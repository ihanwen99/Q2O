import psycopg2
from psycopg2.extras import RealDictCursor

import config


def open_postgres_connection():
    try:
        conn = psycopg2.connect(
            host=config.DB_HOST,
            port=config.DB_PORT,
            database=config.DB_NAME,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
        )
        return conn
    except psycopg2.Error as e:
        print(f"Database Connection Error: {e}")
        return None


def extract_execution_time(explain_analyze_lines):
    execution_time = None
    for line in reversed(explain_analyze_lines):
        if "Execution Time" in line:
            try:
                execution_time_str = line.strip()
                execution_time = float(
                    execution_time_str.split("Execution Time:")[1].strip().split(" ")[0]
                )
                break
            except (IndexError, ValueError) as e:
                print(f"Error extracting execution time: {e}")
                execution_time = None
                break
    return execution_time


def extract_planning_time(explain_analyze_lines):
    for line in explain_analyze_lines:
        if "Planning Time" in line:
            return float(line.split(":")[1].strip().replace("ms", ""))
    return None


def execute_and_explain_query(sql_query):
    conn = open_postgres_connection()
    if conn is None:
        print("Failed to establish a database connection.")
        return None, None, None, None

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(sql_query)
            result = cursor.fetchall()

            explain_query = f"EXPLAIN ANALYZE {sql_query}"
            cursor.execute(explain_query)
            explain_result = cursor.fetchall()

            explain_analyze_lines = [row["QUERY PLAN"] for row in explain_result]
            explain_analyze = "\n".join(explain_analyze_lines)

            execution_time = extract_execution_time(explain_analyze_lines)
            planning_time = extract_planning_time(explain_analyze_lines)

        conn.commit()
        return result, explain_analyze, execution_time, planning_time
    except psycopg2.Error as e:
        print(f"Error executing query: {e}")
        return None, None, None, None
    finally:
        conn.close()
