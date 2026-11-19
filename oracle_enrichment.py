"""Enriquecimiento externo (solo lectura) para run_research en Metaculus.

Agrega dos fuentes de datos duros sin riesgo operativo:
1. Consenso de mercado en Polymarket (solo lectura) como ancla probabilística.
2. Estado oficial semanal de NOAA CPC Niño 3.4 cuando la pregunta toca clima/ENSO.
"""

from __future__ import annotations

import json
import re
from typing import Any

import requests

POLYMARKET_SEARCH_URL = "https://gamma-api.polymarket.com/public-search"
NOAA_NINO_WEEKLY_URL = "https://www.cpc.ncep.noaa.gov/data/indices/wksst9120.for"
HTTP_TIMEOUT_SECONDS = 8.0

_STOPWORDS = frozenset({
    "will", "what", "when", "where", "which", "does", "have", "been", "being",
    "before", "after", "during", "between", "under", "over", "more", "than",
    "less", "least", "most", "this", "that", "these", "those", "with", "from",
    "into", "about", "there", "their", "would", "could", "should", "year",
    "years", "month", "months", "2025", "2026", "2027", "question", "resolve",
})

_CLIMATE_KEYWORDS = frozenset({
    "climate", "weather", "temperature", "hottest", "warming", "hurricane",
    "typhoon", "cyclone", "drought", "rainfall", "snowfall", "enso", "nino",
    "niño", "nina", "niña", "noaa", "pacific", "wildfire", "heatwave",
})

_NOAA_LINE_RE = re.compile(
    r"^\s*(\d{2}[A-Z]{3}\d{4})\s+"
    r"(-?\d+\.\d+)\s*(-?\d+\.\d+)\s+"
    r"(-?\d+\.\d+)\s*(-?\d+\.\d+)\s+"
    r"(-?\d+\.\d+)\s*(-?\d+\.\d+)\s+"
    r"(-?\d+\.\d+)\s*(-?\d+\.\d+)"
)


def extract_query_terms(question_text: str, max_terms: int = 6) -> list[str]:
    """Extrae términos informativos de la pregunta para consultar Polymarket."""
    tokens = re.findall(r"[a-zA-Z0-9]+", question_text.lower())
    seen: set[str] = set()
    terms: list[str] = []
    for tok in tokens:
        if len(tok) < 3 or tok in _STOPWORDS or tok in seen:
            continue
        seen.add(tok)
        terms.append(tok)
        if len(terms) >= max_terms:
            break
    return terms


def is_climate_question(question_text: str) -> bool:
    """Detecta si la pregunta involucra clima, ENSO o fenómenos meteorológicos."""
    lower = question_text.lower()
    return any(keyword in lower for keyword in _CLIMATE_KEYWORDS)


def _overlap_score(query_terms: list[str], title: str) -> float:
    if not query_terms:
        return 0.0
    title_tokens = set(re.findall(r"[a-zA-Z0-9]+", title.lower()))
    hits = sum(1 for term in query_terms if term in title_tokens)
    return hits / len(query_terms)


def _parse_yes_price(raw_prices: Any) -> float | None:
    parsed = raw_prices
    if isinstance(raw_prices, str):
        try:
            parsed = json.loads(raw_prices)
        except ValueError:
            return None
    if isinstance(parsed, list) and parsed:
        try:
            val = float(parsed[0])
            if 0.0 <= val <= 1.0:
                return val
        except (TypeError, ValueError):
            return None
    return None


def fetch_polymarket_prior(
    question_text: str,
    session: requests.Session | None = None,
    min_overlap: float = 0.45,
) -> str:
    """Busca mercados activos equivalentes en Polymarket (solo lectura) como referencia."""
    terms = extract_query_terms(question_text)
    if len(terms) < 2:
        return ""

    client = session or requests.Session()
    try:
        resp = client.get(
            POLYMARKET_SEARCH_URL,
            params={"q": " ".join(terms[:4]), "events_status": "active", "limit_per_type": 6},
            timeout=HTTP_TIMEOUT_SECONDS,
            headers={"User-Agent": "nimbus-metac-bot/0.1"},
        )
        if resp.status_code != 200:
            return ""
        payload = resp.json()
    except (requests.RequestException, ValueError):
        return ""

    events = payload.get("events") if isinstance(payload, dict) else None
    if not isinstance(events, list):
        return ""

    matches: list[tuple[float, str]] = []
    for ev in events:
        if not isinstance(ev, dict) or ev.get("closed"):
            continue
        title = str(ev.get("title") or "").strip()
        score = _overlap_score(terms, title)
        if score < min_overlap:
            continue
        vol = float(ev.get("volume") or 0.0)
        markets = ev.get("markets") if isinstance(ev.get("markets"), list) else []
        price_snippets: list[str] = []
        for m in markets[:3]:
            if not isinstance(m, dict) or m.get("closed"):
                continue
            yes_p = _parse_yes_price(m.get("outcomePrices"))
            if yes_p is None:
                continue
            label = str(m.get("groupItemTitle") or "YES").strip()
            price_snippets.append(f"{label}: {yes_p:.1%}")
        if price_snippets:
            summary = f"- {title} (vol ~${vol:,.0f}): " + " | ".join(price_snippets)
            matches.append((score, summary))

    if not matches:
        return ""

    matches.sort(key=lambda item: item[0], reverse=True)
    lines = [
        "### External Market Prior (Polymarket, read-only reference)",
        *[item[1] for item in matches[:2]],
        "Use this strictly as an external crowd anchor; adjust if resolution criteria differ.",
    ]
    return "\n".join(lines)


def fetch_noaa_enso_context(session: requests.Session | None = None) -> str:
    """Devuelve la última lectura semanal de NOAA CPC Niño 3.4 para preguntas de clima."""
    client = session or requests.Session()
    try:
        resp = client.get(
            NOAA_NINO_WEEKLY_URL,
            timeout=HTTP_TIMEOUT_SECONDS,
            headers={"User-Agent": "nimbus-metac-bot/0.1"},
        )
        if resp.status_code != 200:
            return ""
    except requests.RequestException:
        return ""

    parsed_rows: list[tuple[str, float, float]] = []
    for line in resp.text.splitlines():
        match = _NOAA_LINE_RE.match(line)
        if match:
            parsed_rows.append((match.group(1), float(match.group(6)), float(match.group(7))))

    if not parsed_rows:
        return ""

    week, sst, anomaly = parsed_rows[-1]
    prev_anom = parsed_rows[max(0, len(parsed_rows) - 5)][2]
    delta = anomaly - prev_anom
    return (
        "### Official Climate Observation (NOAA CPC Weekly Niño 3.4)\n"
        f"- Week {week}: Niño 3.4 SST = {sst:.1f}°C, Anomaly = {anomaly:+.1f}°C "
        f"(4-week change: {delta:+.2f}°C)."
    )


def build_external_enrichment(
    question_text: str,
    session: requests.Session | None = None,
) -> str:
    """Construye bloque de contexto externo tolerante a fallos para inyectar en research."""
    blocks: list[str] = []
    prior = fetch_polymarket_prior(question_text, session=session)
    if prior:
        blocks.append(prior)
    if is_climate_question(question_text):
        enso = fetch_noaa_enso_context(session=session)
        if enso:
            blocks.append(enso)
    return "\n\n".join(blocks)
