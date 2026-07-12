import sqlite3
from contextlib import contextmanager

import pandas as pd

from config import DB_PATH


def init_db():
    schema_path = DB_PATH.parent / "schema.sql"
    with sqlite3.connect(DB_PATH) as conn:
        with open(schema_path, "r") as f:
            conn.executescript(f.read())
        conn.commit()


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


def write_df(df: pd.DataFrame, table: str, if_exists: str = "append"):
    with get_conn() as conn:
        df.to_sql(table, conn, if_exists=if_exists, index=False)


def read_sql(query: str, params=None) -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql_query(query, conn, params=params)


def execute(query: str, params=None):
    with get_conn() as conn:
        cur = conn.cursor()
        if params:
            cur.execute(query, params)
        else:
            cur.execute(query)
        conn.commit()