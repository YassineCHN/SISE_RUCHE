import duckdb

con = duckdb.connect("data/local.duckdb")

print(con.execute("SHOW TABLES").fetchall())
con.close()
