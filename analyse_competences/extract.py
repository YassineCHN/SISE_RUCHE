import duckdb
import pandas as pd
from config import MOTHERDUCK_DATABASE

def load_jobs():
    con = duckdb.connect(MOTHERDUCK_DATABASE)

    query = """
    SELECT
        job_id,
        title,
        hard_skills,
        soft_skills
    FROM f_offre
    """
    df = con.execute(query).df()
    con.close()

    # Feature engineering NLP
    df["ml_text"] = (
        df["title"] + " " +
        df["hard_skills"].apply(lambda x: " ".join(x) if x else "") * 2 +
        " " +
        df["soft_skills"].apply(lambda x: " ".join(x) if x else "")
    )

    return df