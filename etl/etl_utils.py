import re
import pandas as pd
from typing import Optional
from datetime import date
import json


def normalize_company_name(value: str) -> str:
    LEGAL_FORMS = [" SA", " SAS", " SASU", " SARL", " EURL", " GIE", " SE"]

    if value is None:
        return "UNKNOWN"

    v = str(value).strip()

    if v == "":
        return "UNKNOWN"

    # Normalisation espaces
    v = re.sub(r"\s+", " ", v)

    # Uppercase pour stabilité analytique
    v = v.upper()

    # Suppression formes juridiques UNIQUEMENT en fin
    for form in LEGAL_FORMS:
        if v.endswith(form):
            v = v[: -len(form)].strip()

    # Nettoyage ponctuation finale
    v = re.sub(r"[,\-–]+$", "", v).strip()

    return v


def normalize_education_level(value: str) -> str:
    """
    Normalize education_level to a small controlled vocabulary.
    Rule: keep the highest level mentioned.
    """
    if value is None:
        return "UNKNOWN"

    v = str(value).strip().lower()

    if v == "" or v in {"nan", "none"}:
        return "UNKNOWN"

    # Aucun prérequis
    if "pas de niveau" in v or "aucun" in v:
        return "AUCUN_PREREQUIS"

    # Ordre IMPORTANT : du plus élevé au plus bas
    if any(k in v for k in ["bac+5", "master", "msc", "grande ecole", "grande école"]):
        return "BAC+5"

    if any(k in v for k in ["bac+4", "bac+3", "licence", "bachelor"]):
        return "BAC+3"

    if any(k in v for k in ["bac+2", "bts", "dut"]):
        return "BAC+2"

    if "bac" in v:
        return "BAC"

    return "UNKNOWN"


def normalize_contract_type(value: str) -> str:
    """
    Normalize raw contract type into 6 categories.
    """
    if not value:
        return "AUTRE"

    v = value.lower()

    if "cdi" in v or "indéterminée" in v:
        return "CDI"

    if "alternance" in v or "apprentissage" in v:
        return "ALTERNANCE"

    if "stage" in v:
        return "STAGE"

    if "intérim" in v or "interim" in v:
        return "INTERIM"

    if "cdd" in v or "durée déterminée" in v:
        return "CDD"

    if (
        "titulaire" in v
        or "contractuel" in v
        or "fonction publique" in v
        or "fonctionnaire" in v
        or "emploi public" in v
        or "territoriale" in v
        or "hospitalière" in v
        or "hospitalier" in v
    ):
        return "CONTRAT_PUBLIC"

    return "AUTRE"


def normalize_start_date(value) -> Optional[str]:
    """
    Normalize start_date into:
    - 'Mois Année'
    - 'DÈS QUE POSSIBLE'
    - NULL
    """
    FRENCH_MONTHS = {
        1: "Janvier",
        2: "Février",
        3: "Mars",
        4: "Avril",
        5: "Mai",
        6: "Juin",
        7: "Juillet",
        8: "Août",
        9: "Septembre",
        10: "Octobre",
        11: "Novembre",
        12: "Décembre",
    }

    if value is None:
        return None

    v = str(value).strip()

    if v == "" or v.lower() in {"nan", "none"}:
        return None

    v_lower = v.lower()

    # ASAP / immédiat
    if any(
        k in v_lower for k in ["dès que", "des que", "asap", "immédiat", "immediat"]
    ):
        return "DÈS QUE POSSIBLE"

    # Date exacte → Mois Année
    try:
        parsed = pd.to_datetime(v, errors="coerce", dayfirst=True)
        if pd.notna(parsed):
            return f"{FRENCH_MONTHS[parsed.month]} {parsed.year}"
    except Exception:
        pass

    # Mois Année déjà présent (Mars 2026, etc.)
    match = re.search(r"([A-Za-zéûôîà]+)\s+(\d{4})", v)
    if match:
        month, year = match.groups()
        return f"{month.capitalize()} {year}"

    return None


def normalize_publication_date(value) -> Optional[date]:
    """
    Normalize publication_date to datetime.date or None
    """
    if value is None:
        return None

    if isinstance(value, pd.Timestamp):
        return value.date()

    value = str(value).strip()

    if value == "" or value.lower() in {"nan", "none"}:
        return None

    try:
        parsed = pd.to_datetime(value, errors="coerce")
        return parsed.date() if pd.notna(parsed) else None
    except Exception:
        return None


def extract_contract_duration_months(value: str) -> Optional[int]:
    """
    Extract contract duration in months from raw text.
    Returns None if not found or ambiguous.
    """
    if not value:
        return None

    v = value.lower()

    # Cas explicite CDI → pas de durée
    if "cdi" in v or "indéterminée" in v:
        return None

    # Exemples : "6 mois", "12 mois", "18 mois"
    m = re.search(r"(\d{1,2})\s*(mois)", v)
    if m:
        return int(m.group(1))

    # Exemples : "1 an", "2 ans"
    m = re.search(r"(\d{1,2})\s*(an|ans)", v)
    if m:
        return int(m.group(1)) * 12

    # Cas flous : renouvelable, permanent, etc.
    return None


def force_list(value):
    """
    Ensure value is always a list.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def serialize_list(value):
    """
    Serialize list-like values into JSON string.
    Preserve structure even in TEXT column.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    if isinstance(value, (list, tuple)):
        clean = [str(v).strip() for v in value if v and str(v).strip()]
        return json.dumps(clean, ensure_ascii=False) if clean else None

    # Si c'est déjà une string (ex: soft_skills déjà aplati)
    return json.dumps([value.strip()], ensure_ascii=False)
