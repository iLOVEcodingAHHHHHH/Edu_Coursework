from pathlib import Path
import duckdb

RAW_DB_PATH = Path(__file__).parent/"datasets.duckdb"
CLEAN_DB_PATH = Path(__file__).parent/"cleansets.duckdb"


# ---clean data pipeline, each table cleaned using seporate file where sql math is applied like the normalization below
# ---AI SLOP AI SLOP AI SLOP AI SLOP AI SLOP AI SLOP AI SLOP AI SLOP AI SLOP ✅ 5. Rule of thumb

# Elementwise math: subtracting means, scaling, multiplying arrays → use Polars, NumPy, PyTorch
# Aggregation, grouping, filtering, joins: use SQL/DuckDB

conn = duckdb.connect("datasets.duckdb")

conn.execute("""
    UPDATE wine
    SET quality = CASE 
                      WHEN quality <= 5 THEN 5
                      ELSE 6
                  END
""")

def create_clean_table(raw_table: str, clean_table: str):
    raw_conn = duckdb.connect(RAW_DB_PATH)
    clean_conn = duckdb.connect(CLEAN_DB_PATH)

    try:
        # Basic cleaning example: remove nulls, normalize a column
        clean_conn.execute(f"""
            CREATE OR REPLACE TABLE {clean_table} AS
            SELECT 
                col1, 
                col2,
                (col3 - AVG(col3) OVER()) / STDDEV(col3) OVER() AS col3_normalized
            FROM raw_conn.{raw_table}
            WHERE col1 IS NOT NULL;
        """)
        print(f"Cleaned table '{clean_table}' created in cleansets.duckdb.")
    finally:
        raw_conn.close()
        clean_conn.close()
# ---AI SLOP AI SLOP AI SLOP AI SLOP AI SLOP AI SLOP AI SLOP AI SLOP AI SLOP 