import psycopg2

def get_connection():
    """
    Create and return one database connection
    """
    try:
        conn = psycopg2.connect(
            host="localhost",
            port="5432",
            database="imdbload",
            user="hanwen",
            password="",
        )
        return conn
    except psycopg2.Error as e:
        print(f"Database Connection Error: {e}")
        return None


if __name__ == '__main__':
    conn = get_connection()
    if conn is None:
        print("Failed to establish a database connection.")