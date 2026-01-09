import duckdb

con = duckdb.connect("data/local.duckdb")
con.execute("SELECT 1")
con.close()
