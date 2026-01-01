"""
NLP léger pour JobTeaser (offline, sans Selenium).
On part du JSON brut (description_raw + champs structurés) et on enrichit.
"""

from __future__ import annotations
import re
from typing import Dict, List, Optional


# ============================================================
# VOCABULAIRES
# ============================================================

LANGUAGES = [
    "français",
    "anglais",
    "espagnol",
    "allemand",
    "italien",
    "portugais",
    "arabe",
    "chinois",
    "japonais",
]

HARD_SKILLS_MAP = {
    "Python": [r"\bpython\b"],
    "SQL": [
        r"\bsql\b",
        r"\bpostgres(ql)?\b",
        r"\bmysql\b",
        r"\bsql server\b",
        r"\boracle\b",
    ],
    "Spark": [r"\bspark\b", r"\bpyspark\b"],
    "Databricks": [r"\bdatabricks\b"],
    "Airflow": [r"\bairflow\b"],
    "dbt": [r"\bdbt\b"],
    "Kafka": [r"\bkafka\b"],
    "Docker": [r"\bdocker\b"],
    "Kubernetes": [r"\bkubernetes\b", r"\bk8s\b"],
    "AWS": [r"\baws\b", r"\bamazon web services\b"],
    "Azure": [r"\bazure\b"],
    "GCP": [r"\bgcp\b", r"\bgoogle cloud\b"],
    "Snowflake": [r"\bsnowflake\b"],
    "BigQuery": [r"\bbigquery\b"],
    "Redshift": [r"\bredshift\b"],
    "TensorFlow": [r"\btensorflow\b"],
    "PyTorch": [r"\bpytorch\b"],
    "scikit-learn": [r"\bscikit[- ]learn\b", r"\bsklearn\b"],
    "Power BI": [r"\bpower ?bi\b"],
    "Tableau": [r"\btableau\b"],
    "Excel": [r"\bexcel\b"],
    "Git": [r"\bgit\b", r"\bgithub\b", r"\bgitlab\b"],
    "CI/CD": [r"\bci/cd\b", r"\bcicd\b"],
    "Linux": [r"\blinux\b"],
    "API": [r"\bapi\b", r"\brest\b"],
    "ML": [r"\bmachine learning\b", r"\bml\b"],
    "NLP": [r"\bnlp\b", r"\btal\b", r"\blangage naturel\b"],
    "LLM": [r"\bllm\b", r"\blarge language model\b", r"\btransformer(s)?\b"],
}

SOFT_SKILLS = {
    "Autonomie": [r"\bautonom(e|ie)\b"],
    "Rigueur": [r"\brigueur\b"],
    "Communication": [r"\bcommunication\b", r"\bcommuniquer\b"],
    "Esprit d'équipe": [r"\besprit d['’]équipe\b", r"\btravail en équipe\b"],
    "Curiosité": [r"\bcuriosit(é|e)\b"],
    "Organisation": [r"\borganis(é|e|ation)\b"],
    "Analyse": [r"\besprit d['’]analyse\b", r"\banalytique\b"],
    "Proactivité": [r"\bproactif\b", r"\bforce de proposition\b"],
}


# ============================================================
# HELPERS
# ============================================================


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _lower(text: str) -> str:
    return _norm(text).lower()


def clean_text_for_nlp(text: str) -> str:
    """
    Nettoyage NLP OFFLINE.
    - AUCUN impact scraping
    - Supprime tous les \\n
    - Texte linéaire, propre, rapide à traiter
    """
    if not text:
        return ""

    text = text.replace("\r", " ").replace("\xa0", " ")
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def _unique_preserve(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# ============================================================
# EXTRACTEURS
# ============================================================


def extract_languages(text: str) -> List[str]:
    t = _lower(text)
    found = []
    for lang in LANGUAGES:
        if re.search(rf"\b{re.escape(lang)}\b", t):
            found.append(lang.capitalize() if lang != "français" else "Français")
    return _unique_preserve(found)


def extract_hard_skills(text: str) -> List[str]:
    t = _lower(text)
    found = []
    for canonical, patterns in HARD_SKILLS_MAP.items():
        for p in patterns:
            if re.search(p, t, flags=re.IGNORECASE):
                found.append(canonical)
                break
    return _unique_preserve(found)


def extract_soft_skills(text: str) -> List[str]:
    t = _lower(text)
    found = []
    for canonical, patterns in SOFT_SKILLS.items():
        for p in patterns:
            if re.search(p, t, flags=re.IGNORECASE):
                found.append(canonical)
                break
    return _unique_preserve(found)


def extract_experience_years(text: str) -> Optional[int]:
    t = _lower(text)

    m = re.search(r"(\d+)\s*(?:à|-)\s*(\d+)\s*an", t)
    if m:
        return min(int(m.group(1)), int(m.group(2)))

    m = re.search(r"(?:minimum|au moins|expérience(?:\s+de)?|exp\.?)\s*(\d+)\s*an", t)
    if m:
        return int(m.group(1))

    m = re.search(r"(\d+)\s*an[s]?\s+d['’]exp", t)
    if m:
        return int(m.group(1))

    return None


def infer_role_family(title: str) -> Optional[str]:
    t = _lower(title)
    if "data engineer" in t or ("engineer" in t and "data" in t):
        return "data_engineer"
    if "data scientist" in t:
        return "data_scientist"
    if "machine learning" in t or re.search(r"\bml\b", t):
        return "ml_engineer"
    if "data analyst" in t or ("analyst" in t and "data" in t):
        return "data_analyst"
    if "consultant" in t and "data" in t:
        return "data_consultant"
    return None


# ============================================================
# PIPELINE NLP
# ============================================================


def apply_jobteaser_nlp(offer: Dict) -> Dict:
    title = offer.get("title", "") or ""

    raw_desc = offer.get("description_raw") or ""
    clean_desc = clean_text_for_nlp(raw_desc)

    text_parts = [
        clean_desc,
        offer.get("function_raw") or "",
        offer.get("education_level_raw") or "",
        offer.get("contract_raw") or "",
        offer.get("remote_raw") or "",
        offer.get("salary_raw") or "",
    ]

    blob = _norm(" ".join(text_parts))

    return {
        "description_clean": clean_desc,
        "content_length": len(clean_desc),
        "role_family": infer_role_family(title),
        "experience_years": extract_experience_years(blob),
        "languages": extract_languages(blob),
        "hard_skills": extract_hard_skills(blob),
        "soft_skills": extract_soft_skills(blob),
    }


def enrich_offers_jobteaser(raw_offers: List[Dict]) -> List[Dict]:
    enriched = []
    total = len(raw_offers)

    for i, offer in enumerate(raw_offers, 1):
        nlp_data = apply_jobteaser_nlp(offer)
        offer_enriched = offer.copy()
        offer_enriched.pop("description_raw", None)  # 🔥 suppression volontaire

        enriched.append({**offer_enriched, **nlp_data, "nlp_success": True})

        if i % 25 == 0 or i == total:
            pct = i / max(1, total) * 100
            print(f"   NLP JobTeaser: {i}/{total} ({pct:.1f}%)", end="\r")

    print()
    return enriched
