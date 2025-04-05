import json
import time
from math import ceil
from math import log

import dimod
from dwave.system import LeapHybridCQMSampler

from backend.normal_execution import execute_query
from backend.utils import generate_table_mapping, generate_postgres_hint, insert_hint_into_sql


def solve_cqm_with_sampler(sampler, cqm: dimod.ConstrainedQuadraticModel, time_limit: int) -> dimod.SampleSet:
    """
    Solves a Constrained Quadratic Model (CQM) using the given sampler.

    Args:
        sampler (LeapHybridCQMSampler): The D-Wave hybrid CQM sampler used to solve the problem.
        cqm (dimod.ConstrainedQuadraticModel): The CQM to solve.

    Returns:
        dimod.SampleSet: A sample set containing the solution.
    """
    start_time = time.time()
    solution = sampler.sample_cqm(cqm, label='Join Order Optimization', time_limit=time_limit)
    solution = solution.filter(lambda row: row.is_feasible)  # Update as feasible solution
    end_time = time.time()
    # print("\t\t=> [Quantum Execution]: sampler.solver", sampler.solver)

    best_sample = solution.first.sample
    best_energy = solution.first.energy
    elapsed_time = end_time - start_time
    # print(f"Quantum Overall Time: {elapsed_time} seconds")

    all_relations = set()
    for key, val in best_sample.items():
        if key.startswith('roj_') and val == 1.0:
            _, relation, _, _ = key.split('_')
            all_relations.add(relation)

    # Dictionary to track the join index for each pair of relations (if any)
    join_pairs = {}
    for key, val in best_sample.items():
        if key.startswith('pred_') and val == 1.0:
            parts = key.split('_')
            rel1, rel2 = parts[1], parts[2]
            join_index = int(parts[-1])
            join_pairs[(rel1, rel2)] = join_index

    #  Sorting join_pairs by join index
    sorted_pairs = sorted(join_pairs.items(), key=lambda x: x[1])

    #  Reconstruct the join order
    final_join_order = []
    for (rel1, rel2), join_index in sorted_pairs:
        if rel1 not in final_join_order:
            final_join_order.append(rel1)
        if rel2 not in final_join_order:
            final_join_order.append(rel2)

    for relation in all_relations:
        if relation not in final_join_order:
            final_join_order.append(relation)
    total_qpu_time = solution.info['qpu_access_time'] / 1e3  # Convert from us to ms

    # print("Final Join Order:", final_join_order)
    # print(f"Cost: {best_energy}")
    # print("Total time in QPU: ", total_qpu_time, "ms")

    return final_join_order, total_qpu_time


def solve_cqm(cqm: dimod.ConstrainedQuadraticModel, time_limit: int) -> dimod.SampleSet:
    """
    Solves a Constrained Quadratic Model (CQM) using the given sampler.

    Args:
        sampler (LeapHybridCQMSampler): The D-Wave hybrid CQM sampler used to solve the problem.
        cqm (dimod.ConstrainedQuadraticModel): The CQM to solve.

    Returns:
        dimod.SampleSet: A sample set containing the solution.
    """
    start_time = time.time()
    sampler = LeapHybridCQMSampler()
    solution = sampler.sample_cqm(cqm, label='Join Order Optimization', time_limit=time_limit)
    solution = solution.filter(lambda row: row.is_feasible)  # Update as feasible solution
    end_time = time.time()
    # print("\t\t=> [Quantum Execution]: sampler.solver", sampler.solver)

    best_sample = solution.first.sample
    best_energy = solution.first.energy
    elapsed_time = end_time - start_time
    # print(f"Quantum Overall Time: {elapsed_time} seconds")

    all_relations = set()
    for key, val in best_sample.items():
        if key.startswith('roj_') and val == 1.0:
            _, relation, _, _ = key.split('_')
            all_relations.add(relation)

    # Dictionary to track the join index for each pair of relations (if any)
    join_pairs = {}
    for key, val in best_sample.items():
        if key.startswith('pred_') and val == 1.0:
            parts = key.split('_')
            rel1, rel2 = parts[1], parts[2]
            join_index = int(parts[-1])
            join_pairs[(rel1, rel2)] = join_index

    #  Sorting join_pairs by join index
    sorted_pairs = sorted(join_pairs.items(), key=lambda x: x[1])

    #  Reconstruct the join order
    final_join_order = []
    for (rel1, rel2), join_index in sorted_pairs:
        if rel1 not in final_join_order:
            final_join_order.append(rel1)
        if rel2 not in final_join_order:
            final_join_order.append(rel2)

    for relation in all_relations:
        if relation not in final_join_order:
            final_join_order.append(relation)
    total_qpu_time = solution.info['qpu_access_time'] / 1e3  # Convert from us to ms

    # print("Final Join Order:", final_join_order)
    # print(f"Cost: {best_energy}")
    # print("Total time in QPU: ", total_qpu_time, "ms")

    return final_join_order, total_qpu_time


def process_input(sql_folder_path):
    cardinalities_file_path = '{}/cardinalities.json'.format(sql_folder_path)
    selectivities_file_path = '{}/selectivities.json'.format(sql_folder_path)
    # Reading the JSON content from the files
    with open(cardinalities_file_path, 'r') as file:
        cardinalities_content = json.load(file)

    with open(selectivities_file_path, 'r') as file:
        selectivities_content = json.load(file)
    return cardinalities_content, selectivities_content

def process_input_demo(sql_folder_path):
    demo_file_path = '{}/demo.json'.format(sql_folder_path)
    # Reading the JSON content from the files
    with open(demo_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    join_order = data["join_order"]
    hint = data["hint"]
    quantum_QUERY = data["quantum_QUERY"]
    return join_order,hint,quantum_QUERY



def build_cqm(cardinalities_content, selectivities_content) -> dimod.ConstrainedQuadraticModel:
    """
    Constructs a Constrained Quadratic Model (CQM) based on the provided parameters.

    Args:
        cardinalities
        selectivities
    Returns:
        dimod.ConstrainedQuadraticModel: The constructed CQM.
    """
    relations = [i for i in range(len(cardinalities_content))]

    # Convert cardinalities to their logarithmic values -> This conversion is important
    log_cardinalities = {relation: round(log(card), 2) for relation, card in zip(relations, cardinalities_content)}

    # Prepare selectivities by checking the selectivity matrix, ensuring uniqueness
    log_selectivities = {}
    for i in range(len(relations)):
        for j in range(i + 1, len(relations)):
            if selectivities_content[i][j] != 1.0:  # If there is a selectivity edge
                log_selectivities[(relations[i], relations[j])] = round(log(selectivities_content[i][j]), 2)

    # Predicates based on log_selectivities keys
    predicates = list(log_selectivities.keys())

    # Initialize the CQM
    cqm = dimod.ConstrainedQuadraticModel()

    # Gather all possible predicates - Corresponds to all joins and rounds of all joins
    # Create binary variables to represent if a predicate is applicable for a join
    pred_vars = {(pred, j): dimod.Binary(f'pred_{pred[0]}_{pred[1]}_join_{j}')
                 for j in range(1, len(relations)) for pred in predicates}
    roj_vars = {(r, j): dimod.Binary(f'roj_{r}_join_{j}') for j in range(1, len(relations) + 1) for r in relations}
    num_joins = len(relations) - 2

    # Constraint 1 for incremental increase of operands
    for r in relations:
        for j in range(1, num_joins):
            # This constraint ensures that if a relation is used in step j, it must continue to be used in step j+1.
            # cqm.add_constraint(roj_vars[(r, j)] <= roj_vars[(r, j + 1)], label=f'cont_use_{r}_{j}')
            cqm.add_constraint(roj_vars[(r, j)] - roj_vars[(r, j + 1)] <= 0, label=f'cont_use_{r}_{j}')

    # Constraint 2 for incremental increase of operands
    for j in range(1, num_joins):
        # This constraint ensures that in step j, exactly j+1 operands participate in the join.
        cqm.add_constraint(sum(roj_vars[(r, j)] for r in relations) == j + 1, label=f'incr_operands_{j}')

    # Add constraints 3 for predicate applicability
    for (pred, j) in pred_vars.keys():
        relation1, relation2 = pred
        # The predicate pred is applicable only if both relations are part of the join
        cqm.add_constraint((pred_vars[(pred, j)] - roj_vars[(relation1, j)]) <= 0, label=f'pred_app_{pred}_{j}_rel1')
        cqm.add_constraint((pred_vars[(pred, j)] - roj_vars[(relation2, j)]) <= 0, label=f'pred_app_{pred}_{j}_rel2')

    max_theta_t = 100  # as per the paper
    N = ceil(log(max_theta_t, 2))  # max_theta_t should be the maximum value theta_t can take

    # binary variables for Buffertj representation for each join j
    stij_vars = {(j, i): dimod.Binary(f'st_{j}_{i}') for j in range(1, len(relations) + 1) for i in range(N)}

    buffertj_expressions = {}
    for j in range(1, len(relations) + 1):
        buffertj_expressions[j] = dimod.quicksum(2 ** i * stij_vars[(j, i)] for i in range(N))

    for j in range(1, len(relations) + 1):
        log_int_card_j_expr = dimod.quicksum(log_cardinalities[r] * roj_vars[(r, j)] for r in relations)

        log_int_card_j_expr += dimod.quicksum(log_selectivities.get((r1, r2), 0) * pred_vars.get(((r1, r2), j), 0)
                                              for r1 in relations for r2 in relations
                                              if r1 != r2 and ((r1, r2), j) in pred_vars)

        buffertj_expr = buffertj_expressions[j]

        penalty_expr = (buffertj_expr - log_int_card_j_expr) ** 2

        # Add penalty expression to CQM objective
        cqm.set_objective(cqm.objective + penalty_expr * max_theta_t)
    return cqm


if __name__ == "__main__":
    CURRENT_QUERY = '''
        SELECT MIN(chn.name) AS uncredited_voiced_character,
               MIN(t.title) AS russian_movie
        FROM char_name AS chn,
             cast_info AS ci,
             company_name AS cn,
             company_type AS ct,
             movie_companies AS mc,
             role_type AS rt,
             title AS t
        WHERE ci.note LIKE '%(voice)%'
          AND ci.note LIKE '%(uncredited)%'
          AND cn.country_code = '[ru]'
          AND rt.role = 'actor'
          AND t.production_year > 2005
          AND t.id = mc.movie_id
          AND t.id = ci.movie_id
          AND ci.movie_id = mc.movie_id
          AND chn.id = ci.person_role_id
          AND rt.id = ci.role_id
          AND cn.id = mc.company_id
          AND ct.id = mc.company_type_id;
        '''

    GLOBAL_PATH = "/tank/users/hanwen/workspace/Quantum4DB"
    sql_folder_path = GLOBAL_PATH + "/benchmark/Demo/JOB/Q1"
    cardinalities_content, selectivities_content = process_input(sql_folder_path)
    print("cardinalities_content: ", cardinalities_content)
    print("selectivities_content: ", selectivities_content)

    cqm = build_cqm(cardinalities_content, selectivities_content)

    # Solve CQM
    sampler = LeapHybridCQMSampler()
    join_order, qpu_time = solve_cqm(cqm)

    mapping = generate_table_mapping(CURRENT_QUERY)
    print(mapping)

    hint = generate_postgres_hint(mapping, join_order)
    quantum_QUERY = insert_hint_into_sql(CURRENT_QUERY, hint)
    print(quantum_QUERY)

    quantum_result, quantum_explain_analyze, quantum_execution_time, quantum_planning_time = execute_query(quantum_QUERY)
