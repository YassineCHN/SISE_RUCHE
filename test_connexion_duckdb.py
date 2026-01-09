import duckdb
from config import load_config

cfg = load_config("dev")
con = duckdb.connect(cfg["duckdb_path"])

print(con.execute("SHOW TABLES").fetchall())
con.close()
