import duckdb
from pathlib import Path
import polars as pl
import marimo as mo

DATASET_DB_PATH = Path(__file__).parent / "datasets.duckdb"

app = mo.App(width="medium")

def list_datasets():
    conn = duckdb.connect(DATASET_DB_PATH)
    tables = [item[0] for item in conn.execute('SHOW TABLES').fetchall()]
    conn.close()
    tables.append('New')
    return tables

def create_table(table_name: str, url: str):
    conn = duckdb.connect(DATASET_DB_PATH)
    conn.execute(f"""
        CREATE TABLE {table_name} AS
        SELECT * FROM read_csv_auto('{url}');
    """)
    conn.close()
    return True

def load_table(table_name: str) -> pl.DataFrame:
    conn = duckdb.connect(DATASET_DB_PATH)
    columnar = conn.execute(f'SELECT * FROM "{table_name}"').arrow()
    conn.close()
    return pl.from_arrow(columnar)

def test(bool):
    if bool:
        return load_table('wine')
    else:
        return new_table_form()
    
def new_table_form():

    return mo.md('''**New Dataset**{url}{name}''').batch(
        url=mo.ui.text(label="URL", full_width=True),
        name=mo.ui.text(label="Set Name:")).form(bordered=False)