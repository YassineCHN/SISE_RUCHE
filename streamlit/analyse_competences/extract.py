import duckdb
import pandas as pd
from streamlit.config import MOTHERDUCK_DATABASE

def load_jobs():
    con = duckdb.connect(md:job_market_RUCHE_final)

    query = """
    SELECT
        job_id,
        title,
        description,
        hard_skills,
        soft_skills
    FROM f_offre
    """
    df = con.execute(query).df()
    con.close()

    # Nettoyage minimal
    df["description"] = (
        df["description"]
        .str.replace("\n", " ", regex=False)
        .str.strip()
    )

    # ---- Construction du texte ML ----
    def build_ml_text(row):
        parts = []

        if row["description"]:
            parts.append(row["description"])

        if row["title"]:
            parts.append(row["title"] * 2)  # pondération légère

        if row["hard_skills"]:
            parts.append(" ".join(row["hard_skills"]) * 3)  # signal fort

        if row["soft_skills"]:
            parts.append(" ".join(row["soft_skills"]))  # signal faible

        return " ".join(parts)

    df["ml_text"] = df.apply(build_ml_text, axis=1)

    return df
