"""Router determinístico de categorías oficiales de Metaculus a 8 playbooks de dominio.

Prioridad de clasificación:
1. Etiquetas oficiales en `question.api_json["projects"]["category"]` (fuente primaria).
2. Reglas por palabras clave en el enunciado (respaldo si la pregunta no trae categoría).
3. `general_fallback` (garantiza cobertura 100% sin excepciones ni categorías huérfanas).
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

# Orden de prioridad cuando una pregunta tiene múltiples etiquetas oficiales de Metaculus
PLAYBOOK_PRIORITY: tuple[str, ...] = (
    "climate_environment",
    "crypto",
    "ai_tech_space",
    "health_science",
    "sports_entertainment",
    "economy_business",
    "politics_geo_law",
    "general_fallback",
)

# Mapa exhaustivo de categorías oficiales de Metaculus (normalizadas en minúsculas) -> playbook
OFFICIAL_CATEGORY_TO_PLAYBOOK: dict[str, str] = {
    # 1. Política, Elecciones, Geopolítica, Justicia y Riesgo Nuclear (~42%)
    "elections": "politics_geo_law",
    "geopolitics": "politics_geo_law",
    "politics": "politics_geo_law",
    "law": "politics_geo_law",
    "nuclear technology & risks": "politics_geo_law",
    "security & defense": "politics_geo_law",
    "global governance": "politics_geo_law",
    # 2. Economía y Negocios (~27%)
    "economy & business": "economy_business",
    "economics": "economy_business",
    "business": "economy_business",
    "finance": "economy_business",
    # 3. Salud y Ciencias Naturales (~11%)
    "health & pandemics": "health_science",
    "natural sciences": "health_science",
    "biology": "health_science",
    "medicine": "health_science",
    "social sciences": "health_science",
    # 4. Deportes y Entretenimiento (~9%)
    "sports & entertainment": "sports_entertainment",
    "sports": "sports_entertainment",
    "entertainment": "sports_entertainment",
    "culture": "sports_entertainment",
    # 5. Clima y Ambiente (~6%)
    "environment & climate": "climate_environment",
    "climate": "climate_environment",
    "energy": "climate_environment",
    # 6. Criptomonedas (~4%)
    "cryptocurrencies": "crypto",
    "crypto": "crypto",
    # 7. IA, Tecnología, Computación y Espacio (~5%)
    "artificial intelligence": "ai_tech_space",
    "technology": "ai_tech_space",
    "computing and math": "ai_tech_space",
    "space": "ai_tech_space",
}

# Reglas de respaldo cuando la pregunta no tiene metadata oficial de categoría
_KEYWORD_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("climate_environment", ("climate", "el nino", "el niño", "la nina", "hurricane", "temperature", "noaa", "warming", "wildfire")),
    ("crypto", ("bitcoin", "ethereum", "btc", "eth", "solana", "crypto", "stablecoin", "defi")),
    ("ai_tech_space", ("openai", "anthropic", "gemini", "gpt", "llm", "artificial intelligence", "spacex", "nasa", "semiconductor", "nvidia")),
    ("health_science", ("fda", "cdc", "who", "pandemic", "vaccine", "measles", "virus", "clinical trial", "disease")),
    ("sports_entertainment", ("nba", "nfl", "fifa", "world cup", "olympics", "oscar", "box office", "grammy", "super bowl", "champions league")),
    ("economy_business", ("fed", "federal reserve", "inflation", "cpi", "gdp", "s&p 500", "nasdaq", "unemployment", "interest rate", "recession", "stock", "revenue", "outlays")),
    ("politics_geo_law", ("election", "president", "senate", "congress", "supreme court", "lawsuit", "indicted", "ceasefire", "nato", "tariff", "sanctions", "parliament")),
)


@dataclass(frozen=True)
class DomainRoute:
    """Resultado de clasificación de una pregunta de Metaculus."""

    playbook: str
    official_categories: tuple[str, ...]
    matched_by: str  # "official_category", "keyword_fallback", "default_fallback"
    question_type: str


def normalize_category_name(name: str) -> str:
    """Limpia caracteres invisibles (ej. soft-hyphen en 'Crypto\\xadcurrencies') y espacios."""
    cleaned = name.replace("\u00ad", "").replace("\u200b", "").strip().lower()
    return re.sub(r"\s+", " ", cleaned)


def extract_official_categories(api_json: dict[str, Any] | None) -> tuple[str, ...]:
    """Extrae los nombres normalizados de categorías desde el payload de Metaculus."""
    if not isinstance(api_json, dict):
        return ()
    projects = api_json.get("projects")
    if not isinstance(projects, dict):
        return ()
    raw_cats = projects.get("category")
    if not isinstance(raw_cats, list):
        return ()

    result: list[str] = []
    for entry in raw_cats:
        if isinstance(entry, dict):
            raw_name = str(entry.get("name") or entry.get("slug") or "")
        else:
            raw_name = str(entry or "")
        norm = normalize_category_name(raw_name)
        if norm and norm not in result:
            result.append(norm)
    return tuple(result)


def classify_question_data(
    question_text: str,
    api_json: dict[str, Any] | None = None,
    question_type: str = "unknown",
) -> DomainRoute:
    """Clasifica una pregunta de Metaculus a uno de los 8 playbooks sin dejar huecos."""
    official_cats = extract_official_categories(api_json)

    matched_playbooks: set[str] = set()
    for cat in official_cats:
        if cat in OFFICIAL_CATEGORY_TO_PLAYBOOK:
            matched_playbooks.add(OFFICIAL_CATEGORY_TO_PLAYBOOK[cat])

    if matched_playbooks:
        for pb in PLAYBOOK_PRIORITY:
            if pb in matched_playbooks:
                return DomainRoute(
                    playbook=pb,
                    official_categories=official_cats,
                    matched_by="official_category",
                    question_type=question_type,
                )

    lower_text = question_text.lower()
    for pb, keywords in _KEYWORD_RULES:
        if any(kw in lower_text for kw in keywords):
            return DomainRoute(
                playbook=pb,
                official_categories=official_cats,
                matched_by="keyword_fallback",
                question_type=question_type,
            )

    return DomainRoute(
        playbook="general_fallback",
        official_categories=official_cats,
        matched_by="default_fallback",
        question_type=question_type,
    )


def classify_metaculus_question(question: Any) -> DomainRoute:
    """Atajo para clasificar objetos `MetaculusQuestion` del SDK forecasting-tools."""
    return classify_question_data(
        question_text=str(getattr(question, "question_text", "") or ""),
        api_json=getattr(question, "api_json", None),
        question_type=type(question).__name__,
    )
