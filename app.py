import math
import os

from dwave.system import LeapHybridCQMSampler
from flask import Flask, render_template, request, jsonify, send_from_directory
from natsort import natsorted

from backend.nl import (
    sample_nl_model,
    build_left_deep_nl_model,
    read_nl_join_order,
    create_nl_sampler,
)
from backend.postgres import execute_and_explain_query
from backend.cqm import build_cqm, load_statistics, solve_cqm_with_sampler
from backend.utils import (
    read_query_sql,
    build_table_alias_mapping,
    build_postgres_hint,
    prepend_hint_to_sql,
    compute_db_cost,
)

import config

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(ROOT_DIR, "frontend"))
sampler = LeapHybridCQMSampler()

BENCHMARK_PATH = os.path.join(ROOT_DIR, "benchmark")
SQL_DIR = os.path.join(BENCHMARK_PATH, "JOB")
SYNTHETIC_PATH = os.path.join(ROOT_DIR, "benchmark", "Synthetic")

CURRENT_QUERY = ""
CURRENT_SQL_FOLDER = ""

QUANTUM_TIME_LIMIT = 5

quantum_explain_analyze_output = None
postgres_explain_analyze_output = None


def list_subdirectories(path):
    return natsorted(
        d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))
    )


@app.route("/")
def index():
    synthetic_sizes = {
        scheme: [
            int(d.removesuffix("relations"))
            for d in list_subdirectories(os.path.join(SYNTHETIC_PATH, scheme))
        ]
        for scheme in list_subdirectories(SYNTHETIC_PATH)
    }
    return render_template(
        "index.html",
        benchmarks=["JOB"],
        files=list_subdirectories(SQL_DIR),
        synthetic_sizes=synthetic_sizes,
    )


@app.route("/fetch-sql", methods=["GET"])
def fetch_sql():
    global CURRENT_QUERY
    global CURRENT_SQL_FOLDER

    benchmark = request.args.get("benchmark")
    query_group = request.args.get("query_group")

    if not benchmark or not query_group:
        return jsonify({"sql_content": "Invalid benchmark or query group provided."})

    CURRENT_SQL_FOLDER = os.path.join(BENCHMARK_PATH, benchmark, query_group)
    CURRENT_QUERY = read_query_sql(CURRENT_SQL_FOLDER)
    return jsonify({"sql_content": CURRENT_QUERY})


@app.route("/run-quantum", methods=["POST"])
def run_cqm_query():
    cardinalities_content, selectivities_content = load_statistics(CURRENT_SQL_FOLDER)
    cqm = build_cqm(cardinalities_content, selectivities_content)
    join_order, qpu_time = solve_cqm_with_sampler(
        sampler, cqm, time_limit=QUANTUM_TIME_LIMIT
    )

    mapping = build_table_alias_mapping(CURRENT_QUERY)
    hint = build_postgres_hint(mapping, join_order)
    quantum_QUERY = prepend_hint_to_sql(CURRENT_QUERY, hint)

    (
        quantum_result,
        quantum_explain_analyze,
        quantum_execution_time,
        quantum_planning_time,
    ) = execute_and_explain_query(quantum_QUERY)

    global quantum_explain_analyze_output
    quantum_explain_analyze_output = quantum_explain_analyze

    join_order = join_order if join_order is not None else []
    quantum_explain_analyze = (
        quantum_explain_analyze if quantum_explain_analyze is not None else "N/A"
    )
    quantum_result = quantum_result if quantum_result is not None else "N/A"
    qpu_time = qpu_time if isinstance(qpu_time, (int, float)) else 0
    quantum_planning_time = (
        quantum_planning_time if isinstance(quantum_planning_time, (int, float)) else 0
    )
    quantum_execution_time = (
        quantum_execution_time
        if isinstance(quantum_execution_time, (int, float))
        else "N/A"
    )

    output_result = (
        "=> Join Order: "
        + " ".join(join_order)
        + "\n"
        + "=> Quantum Result: {}\n\n\n".format(quantum_result)
        + "=> Quantum Planning Time: {:.3f} ms\n".format(qpu_time)
        + "=> Quantum-Postgres Plan Time: {:.3f} ms\n".format(quantum_planning_time)
        + "=> DB Execution Time: {} ms".format(quantum_execution_time)
    )

    return jsonify(
        {
            "result": output_result,
            "quantumQueryResult": quantum_result,
            "joinOrder": " ".join(join_order),
            "joinOrderHint": hint,
            "quantumPlanningTime": qpu_time,
            "quantumPostgresPlanTime": quantum_planning_time,
            "quantumExecutionTime": quantum_execution_time,
        }
    )


@app.route("/run-default", methods=["POST"])
def run_postgres_query():
    result, explain_analyze, execution_time, planning_time = execute_and_explain_query(
        CURRENT_QUERY
    )
    global postgres_explain_analyze_output
    postgres_explain_analyze_output = explain_analyze

    output_result = (
        "=> Result: {}\n\n\n".format(result)
        + "=> Postgres Plan Time: {:.3f} ms\n".format(planning_time)
        + "=> DB Execution Time: {} ms".format(execution_time)
    )

    return jsonify(
        {
            "result": output_result,
            "postgresQueryResult": result,
            "postgresPlanningTime": planning_time,
            "postgresExecutionTime": execution_time,
        }
    )


@app.route("/visualize-plan", methods=["POST"])
def fetch_query_plan():
    global postgres_explain_analyze_output, quantum_explain_analyze_output

    plan_type = request.form["type"]

    if plan_type == "quantum":
        query_plan_details = quantum_explain_analyze_output
    elif plan_type == "postgres":
        query_plan_details = postgres_explain_analyze_output
    else:
        return jsonify({"error": "Invalid plan type provided."}), 400

    if not query_plan_details:
        return jsonify({"error": "No plan available for visualization."}), 400

    return jsonify({"plan": query_plan_details})


@app.route("/process-quantum-manipulation", methods=["POST"])
def run_nl_solver():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid request data"}), 400

    scheme = data.get("scheme")
    query_id = data.get("query_id")
    annealing_time = data.get("annealing_time")
    relations = data.get("relations")

    synthetic_query_folder = os.path.join(
        SYNTHETIC_PATH,
        str(scheme),
        "{}relations".format(relations),
        str(int(query_id) - 1),
    )
    cardinalities_content, selectivities_content = load_statistics(
        synthetic_query_folder
    )

    nl_model, nl_query_optimization_time = build_left_deep_nl_model(
        cardinalities_content, selectivities_content
    )

    nl_sampler, nl_initialize_sampler_time = create_nl_sampler()
    nl_solution, nl_solver_sample_time = sample_nl_model(
        nl_sampler, nl_model, annealing_time
    )
    nl_join_order, nl_fetch_join_order_time = read_nl_join_order(nl_model)
    nl_list_join_order = nl_join_order.tolist()

    db_cost = compute_db_cost(
        nl_join_order, cardinalities_content, selectivities_content
    )
    if math.isinf(db_cost):
        db_cost = "Inf"
    return jsonify(
        {
            "message": "Parameters received successfully!",
            "relations": relations,
            "annealing_time": annealing_time,
            "db_cost": db_cost,
            "nl_join_order": nl_list_join_order,
            "nl_query_optimization_time": nl_query_optimization_time,
            "nl_initialize_sampler_time": nl_initialize_sampler_time,
            "nl_solver_sample_time": nl_solver_sample_time,
            "nl_fetch_join_order_time": nl_fetch_join_order_time,
        }
    )


app.config["UPLOAD_FOLDER"] = os.path.join(ROOT_DIR, "frontend", "assets")


@app.route("/assets/<filename>")
def serve_asset(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG, use_reloader=False)
