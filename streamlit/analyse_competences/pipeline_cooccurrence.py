import duckdb
import itertools
import pandas as pd
from collections import Counter

con = duckdb.connect("md:job_market_RUCHE")

df = con.execute(
    """
    SELECT job_id, hard_skills
    FROM f_offre
"""
).df()

pairs = Counter()
for skills in df["hard_skills"]:
    if skills and len(skills) > 1:
        for a, b in itertools.combinations(sorted(set(skills)), 2):
            pairs[(a, b)] += 1

graph_df = pd.DataFrame(
    [(a, b, w) for (a, b), w in pairs.items() if w >= 5],
    columns=["skill_1", "skill_2", "weight"],
)

con.execute("DROP TABLE IF EXISTS skill_cooccurrence")
con.execute("CREATE TABLE skill_cooccurrence AS SELECT * FROM graph_df")
con.close()
