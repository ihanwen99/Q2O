import math
import os

from dwave.system import LeapHybridCQMSampler
from flask import Flask, render_template, request, jsonify, send_from_directory
from natsort import natsorted

from backend.nl_solver_execution import nl_solver_sample, nl_query_optimization, nl_fetch_join_order, nl_initialize_sampler
from backend.normal_execution import execute_query
from backend.quantum_execution import build_cqm, process_input, solve_cqm_with_sampler, process_input_demo
from backend.utils import fetchSQLContent, generate_table_mapping, generate_postgres_hint, insert_hint_into_sql, compute_db_cost

app = Flask(__name__, template_folder='frontend')
sampler = LeapHybridCQMSampler()

# Set the directory path for SQL files
BENCHMARK_PATH = "benchmark/Demo"
SQL_DIR = "{}/JOB".format(BENCHMARK_PATH)
SYNTHETIC_PATH = "benchmark/Synthetic"

CURRENT_QUERY = ""
CURRENT_SQL_FOLDER = ""

QUANTUM_TIME_LIMIT = 5

# Store EXPLAIN ANALYZE results for visualization
quantum_explain_analyze_output = None
postgres_explain_analyze_output = None

# Folder to save visualizations
VISUALIZATION_FOLDER = 'static/visualizations'
os.makedirs(VISUALIZATION_FOLDER, exist_ok=True)


# Retrieve available benchmarks and SQL files
@app.route('/')
def index():
    # Get the list of directories as benchmarks
    benchmarks = ["JOB", "LDBC", "SQLite", "TPC-H", "TPC-DS"]
    directories = [d for d in os.listdir(SQL_DIR) if os.path.isdir(os.path.join(SQL_DIR, d))]
    directories = natsorted(directories)
    # Get the list of SQL files
    files = [f for f in os.listdir(SQL_DIR) if f.endswith('.sql')]
    return render_template('index.html', benchmarks=benchmarks, files=directories)


# Fetch the content of the specified SQL file
@app.route('/fetch-sql', methods=['GET'])
def fetch_sql():
    global CURRENT_QUERY
    global CURRENT_SQL_FOLDER

    benchmark = request.args.get('benchmark')
    query_group = request.args.get('query_group')

    # Ensure valid benchmark and query_group parameters are provided
    if not benchmark or not query_group:
        return jsonify({'sql_content': "Invalid benchmark or query group provided."})

    # Build the SQL file path
    CURRENT_SQL_FOLDER = "{}/{}/{}/".format(BENCHMARK_PATH, benchmark, query_group)
    CURRENT_QUERY = fetchSQLContent(CURRENT_SQL_FOLDER)
    return jsonify({'sql_content': CURRENT_QUERY})


# Execute the query and retrieve the execution plan
@app.route('/run', methods=['POST'])
def run_query():
    selected_file = request.form['sql_file']
    # Simulate execution
    result = f"Processed: {selected_file}"
    return jsonify({'result': result})


# Quantum plans and results
@app.route('/run-quantum', methods=['POST'])
def run_quantum():
    cardinalities_content, selectivities_content = process_input(CURRENT_SQL_FOLDER)
    cqm = build_cqm(cardinalities_content, selectivities_content)
    join_order, qpu_time = solve_cqm_with_sampler(sampler, cqm, time_limit=QUANTUM_TIME_LIMIT)

    mapping = generate_table_mapping(CURRENT_QUERY)
    hint = generate_postgres_hint(mapping, join_order)
    quantum_QUERY = insert_hint_into_sql(CURRENT_QUERY, hint)

    join_order, hint, quantum_QUERY = process_input_demo(CURRENT_SQL_FOLDER)
    print(quantum_QUERY)

    quantum_result, quantum_explain_analyze, quantum_execution_time, quantum_planning_time = (
        execute_query(quantum_QUERY))

    # Save the EXPLAIN ANALYZE result for visualization
    global quantum_explain_analyze_output  # Access the global variable
    quantum_explain_analyze_output = quantum_explain_analyze

    join_order = join_order if join_order is not None else []
    quantum_explain_analyze = quantum_explain_analyze if quantum_explain_analyze is not None else "N/A"
    quantum_result = quantum_result if quantum_result is not None else "N/A"
    # elapsed_time = elapsed_time if isinstance(elapsed_time, (int, float)) else 0
    qpu_time = qpu_time if isinstance(qpu_time, (int, float)) else 0
    quantum_planning_time = quantum_planning_time if isinstance(quantum_planning_time, (int, float)) else 0
    quantum_execution_time = quantum_execution_time if isinstance(quantum_execution_time, (int, float)) else "N/A"

    # Build the output string
    output_result = (
            "=> Join Order: " + " ".join(join_order) + "\n" +
            # "=> Quantum Plan: \n{}\n\n".format(quantum_explain_analyze) +
            "=> Quantum Result: {}\n\n\n".format(quantum_result) +
            # "=> Quantum Compute Time: {:.3f} ms\n".format(elapsed_time) +
            "=> Quantum Planning Time: {:.3f} ms\n".format(qpu_time) +
            "=> Quantum-Postgres Plan Time: {:.3f} ms\n".format(quantum_planning_time) +
            "=> DB Execution Time: {} ms".format(quantum_execution_time))

    return jsonify({'result': output_result,
                    'quantumQueryResult': quantum_result,
                    'joinOrder': join_order,
                    'joinOrderHint': hint,
                    'quantumPlanningTime': qpu_time,
                    'quantumPostgresPlanTime': quantum_planning_time,
                    'quantumExecutionTime': quantum_execution_time,
                    })


# Postgres plans and results
@app.route('/run-default', methods=['POST'])
def run_default():
    result, explain_analyze, execution_time, planning_time = execute_query(CURRENT_QUERY)
    # Save the EXPLAIN ANALYZE result for visualization
    global postgres_explain_analyze_output  # Access the global variable
    postgres_explain_analyze_output = explain_analyze

    output_result = (
            "=> Result: {}\n\n\n".format(result) +
            # "=> Plan: \n{}\n\n".format(explain_analyze) +
            "=> Postgres Plan Time: {:.3f} ms\n".format(planning_time) +
            "=> DB Execution Time: {} ms".format(execution_time))

    return jsonify({'result': output_result,
                    'postgresQueryResult': result,
                    'postgresPlanningTime': planning_time,
                    'postgresExecutionTime': execution_time,
                    })


@app.route('/visualize-plan', methods=['POST'])
def visualize_plan():
    global postgres_explain_analyze_output, quantum_explain_analyze_output  # Access the global variables

    plan_type = request.form['type']
    print("Plan type requested:", plan_type)

    if plan_type == "quantum":
        query_plan_details = quantum_explain_analyze_output
    elif plan_type == "postgres":
        query_plan_details = postgres_explain_analyze_output
    else:
        return jsonify({'error': 'Invalid plan type provided.'}), 400

    if not query_plan_details:
        return jsonify({'error': 'No plan available for visualization.'}), 400

    print("Returning query plan:", query_plan_details)
    return jsonify({'plan': query_plan_details})


@app.route('/process-quantum-manipulation', methods=['POST'])
def process_quantum_mul():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid request data'}), 400

    # Extract parameters
    scheme = data.get('scheme')
    query_id = data.get('query_id')
    annealing_time = data.get('annealing_time')
    relations = data.get('relations')
    print(f"Received parameters - Scheme: {scheme}, Query ID: {query_id}, Annealing Time: {annealing_time}, #Relations: {relations}")

    synthetic_query_folder = "{}/{}/{}relations/{}".format(SYNTHETIC_PATH, scheme, relations, int(query_id) - 1)
    cardinalities_content, selectivities_content = process_input(synthetic_query_folder)

    nl_model, nl_query_optimization_time = nl_query_optimization(cardinalities_content, selectivities_content)
    # print(dir(nl_model.states))
    print("nl_model: ", nl_model, f"{nl_query_optimization_time:.3f} seconds")
    # print("nl_model => num_states: ", len(nl_model.states))

    nl_sampler, nl_initialize_sampler_time = nl_initialize_sampler()
    # print("nl_sampler => estimated_min_time_limit: ", nl_sampler.estimated_min_time_limit(nl_model))
    # print("nl_sampler => max_num_states: ", nl_sampler.solver.properties.get("maximum_number_of_states"))
    print("nl_sampler: ", nl_sampler, f"{nl_initialize_sampler_time:.3f} seconds")
    nl_solution, nl_solver_sample_time = nl_solver_sample(nl_sampler, nl_model, annealing_time)
    print("nl_solution: ", nl_solution, f"{nl_solver_sample_time:.3f} seconds")
    nl_join_order, nl_fetch_join_order_time = nl_fetch_join_order(nl_model)
    nl_list_join_order = nl_join_order.tolist()
    print("nl_join_order: ", nl_list_join_order, f"{nl_fetch_join_order_time:.3f} seconds")

    db_cost = compute_db_cost(nl_join_order, cardinalities_content, selectivities_content)
    print(db_cost)
    if math.isinf(db_cost): db_cost = "Inf"
    return jsonify({'message': 'Parameters received successfully!',
                    'relations': relations,
                    'annealing_time': annealing_time,
                    'db_cost': db_cost,
                    'nl_join_order': nl_list_join_order,
                    'nl_query_optimization_time': nl_query_optimization_time,
                    'nl_initialize_sampler_time': nl_initialize_sampler_time,
                    'nl_solver_sample_time': nl_solver_sample_time,
                    'nl_fetch_join_order_time': nl_fetch_join_order_time,
                    })


app.config['UPLOAD_FOLDER'] = os.path.join('frontend', 'assets')


@app.route('/assets/<filename>')
def serve_static(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)
