import glob
import os
import time

import sqlparse
from sqlparse.sql import IdentifierList, Identifier
from sqlparse.tokens import Keyword


def read_query_sql(sql_folder_path):
    sql_files = glob.glob(os.path.join(sql_folder_path, "*.sql"))

    if not sql_files:
        print(f"No .sql file found in {sql_folder_path}")
        return

    sql_file_path = sql_files[0]

    try:
        with open(sql_file_path, "r", encoding="utf-8") as file:
            sql_content = file.read()
        return sql_content
    except Exception as e:
        print(f"Error reading SQL file {sql_file_path}: {e}")


def build_table_alias_mapping(sql_query):
    parsed = sqlparse.parse(sql_query)[0]
    from_seen = False
    tables = []

    for token in parsed.tokens:
        if from_seen:
            if isinstance(token, IdentifierList):
                for identifier in token.get_identifiers():
                    alias = identifier.get_alias()
                    if alias:
                        tables.append(alias)
            elif isinstance(token, Identifier):
                alias = token.get_alias()
                if alias:
                    tables.append(alias)
            elif token.ttype is Keyword and token.value.upper() in (
                "WHERE",
                "GROUP",
                "ORDER",
                "HAVING",
                "LIMIT",
            ):
                break
        elif token.ttype is Keyword and token.value.upper() == "FROM":
            from_seen = True

    mapping = {i: table for i, table in enumerate(tables)}
    return mapping


def build_postgres_hint(mapping, join_order):
    table_aliases = [mapping[int(i)] for i in join_order]
    hint = "/*+\n    Leading (" + " ".join(table_aliases) + ")\n*/"
    return hint


def prepend_hint_to_sql(sql_query, hint):
    return hint + "\n" + sql_query


def compute_db_cost(join_order: list, cardinalities, selectivities):
    total_cost = 0

    num_relations = len(join_order)
    curr_rid = join_order[0]
    intermediate_cardinality = cardinalities[curr_rid]
    involved_relations = [curr_rid]

    for i in range(1, num_relations):
        right_rid = join_order[i]
        current_selectivity = 1
        for left_rid in involved_relations:
            if abs(selectivities[left_rid][right_rid] - 1) == 0.00000001:
                return "JOIN UNPROPER RELATIONS..."
            current_selectivity *= selectivities[left_rid][right_rid]
        intermediate_cardinality = (
            intermediate_cardinality * cardinalities[right_rid] * current_selectivity
        )
        total_cost += intermediate_cardinality
        involved_relations.append(right_rid)

    return total_cost


def list_folders_by_number(directory):
    all_items = os.listdir(directory)
    folders = [
        item for item in all_items if os.path.isdir(os.path.join(directory, item))
    ]
    sorted_folders = sorted(
        folders,
        key=lambda x: (
            int("".join(filter(str.isdigit, x)))
            if any(c.isdigit() for c in x)
            else float("inf")
        ),
    )
    return [os.path.join(directory, folder) for folder in sorted_folders]


def return_result_and_elapsed_time(func):
    def timed_call(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return result, execution_time

    return timed_call
