"""Conectores abiertos científicos y biomédicos (0 tokens LLM, sin API key).

Fuentes verificadas:
- arXiv Atom API: verifica en vivo preprints publicados por autores o tema.
- ClinicalTrials.gov API v2: conteo real y estado de ensayos clínicos.
- Google News RSS multilingüe: soporta consultas locales (AR/BR/EN) para brotes
  y eventos regionales (ej. InfoGripe Brasil, hospitales en Argentina).
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

import requests

from playbooks.open_sources import (
    DEFAULT_HEADERS,
    HTTP_TIMEOUT_SECONDS,
    NewsSnippet,
    _BLOCKED_NEWS_SOURCES,
    _extract_focused_terms,
    _format_pub_date,
    _overlap_score,
)

ARXIV_API_URL = "https://export.arxiv.org/api/query"
CLINICAL_TRIALS_URL = "https://clinicaltrials.gov/api/v2/studies"
GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"

_ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}
_LOCALE_HINTS: tuple[tuple[tuple[str, ...], str, str, str], ...] = (
    (("mendoza", "argentina", "buenos aires", "córdoba"), "es-419", "AR", "AR:es-419"),
    (("brazil", "brasil", "infogripe", "fiocruz", "srag"), "pt-BR", "BR", "BR:pt-419"),
    (("mexico", "méxico", "senasica"), "es-419", "MX", "MX:es-419"),
)


@dataclass(frozen=True)
class ArxivPaper:
    """Preprint recuperado desde la API abierta de arXiv."""

    published: str
    title: str
    authors: tuple[str, ...]


@dataclass(frozen=True)
class ClinicalTrialsSummary:
    """Resumen de búsqueda en ClinicalTrials.gov API v2."""

    query: str
    total_count: int
    sample_titles: tuple[str, ...]


def fetch_arxiv_preprints(
    question_text: str,
    *,
    session: requests.Session | None = None,
    max_results: int = 3,
) -> list[ArxivPaper]:
    """Busca preprints relevantes en arXiv por términos clave/autores."""
    terms = _extract_focused_terms(question_text, max_terms=5)
    if len(terms) < 2:
        return []

    query = " AND ".join(f"all:{t}" for t in terms[:4])
    client = session or requests.Session()
    try:
        resp = client.get(
            ARXIV_API_URL,
            params={
                "search_query": query,
                "start": 0,
                "max_results": max_results,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            },
            headers=DEFAULT_HEADERS,
            timeout=HTTP_TIMEOUT_SECONDS,
        )
        if resp.status_code != 200:
            return []
        root = ET.fromstring(resp.text)
    except (requests.RequestException, ET.ParseError):
        return []

    papers: list[ArxivPaper] = []
    for entry in root.findall("atom:entry", _ARXIV_NS):
        title = " ".join((entry.findtext("atom:title", default="", namespaces=_ARXIV_NS)).split())
        published = (entry.findtext("atom:published", default="", namespaces=_ARXIV_NS))[:10]
        authors = tuple(
            (a.findtext("atom:name", default="", namespaces=_ARXIV_NS)).strip()
            for a in entry.findall("atom:author", _ARXIV_NS)[:3]
        )
        if title:
            papers.append(ArxivPaper(published=published or "unknown", title=title[:180], authors=authors))
    return papers


def fetch_clinical_trials_summary(
    question_text: str,
    *,
    session: requests.Session | None = None,
    recruiting_only: bool = False,
) -> ClinicalTrialsSummary | None:
    """Consulta ClinicalTrials.gov API v2 para contar estudios y listar muestras."""
    terms = _extract_focused_terms(question_text, max_terms=4)
    if len(terms) < 2:
        return None

    query_str = " ".join(terms[:3])
    params: dict[str, str | int] = {
        "query.term": query_str,
        "countTotal": "true",
        "pageSize": 3,
    }
    if recruiting_only:
        params["filter.overallStatus"] = "RECRUITING"

    client = session or requests.Session()
    try:
        resp = client.get(
            CLINICAL_TRIALS_URL,
            params=params,
            headers=DEFAULT_HEADERS,
            timeout=HTTP_TIMEOUT_SECONDS,
        )
        if resp.status_code != 200:
            return None
        payload = resp.json()
    except (requests.RequestException, ValueError):
        return None

    if not isinstance(payload, dict):
        return None

    total = int(payload.get("totalCount") or 0)
    studies = payload.get("studies") if isinstance(payload.get("studies"), list) else []
    titles: list[str] = []
    for st in studies[:3]:
        proto = st.get("protocolSection") if isinstance(st, dict) else None
        ident = proto.get("identificationModule") if isinstance(proto, dict) else None
        title = str((ident or {}).get("briefTitle") or "").strip()
        if title:
            titles.append(title[:160])

    return ClinicalTrialsSummary(
        query=query_str,
        total_count=total,
        sample_titles=tuple(titles),
    )


def detect_regional_locale(question_text: str) -> tuple[str, str, str] | None:
    """Detecta si la pregunta corresponde a una región no angloparlante soportada."""
    lower = question_text.lower()
    for keywords, hl, gl, ceid in _LOCALE_HINTS:
        if any(kw in lower for kw in keywords):
            return hl, gl, ceid
    return None


def fetch_regional_news_rss(
    question_text: str,
    *,
    session: requests.Session | None = None,
    max_items: int = 4,
) -> list[NewsSnippet]:
    """Consulta Google News RSS en el idioma/país regional si la pregunta es local."""
    locale = detect_regional_locale(question_text)
    if locale is None:
        return []

    hl, gl, ceid = locale
    terms = _extract_focused_terms(question_text, max_terms=5)
    if len(terms) < 2:
        return []

    query = f"{' '.join(terms[:4])} when:60d"
    url = f"{GOOGLE_NEWS_RSS_URL}?q={quote_plus(query)}&hl={hl}&gl={gl}&ceid={ceid}"
    client = session or requests.Session()
    try:
        resp = client.get(url, headers=DEFAULT_HEADERS, timeout=HTTP_TIMEOUT_SECONDS)
        if resp.status_code != 200:
            return []
        root = ET.fromstring(resp.text)
    except (requests.RequestException, ET.ParseError):
        return []

    snippets: list[NewsSnippet] = []
    seen: set[str] = set()
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        if not title or title.lower()[:80] in seen:
            continue
        if _overlap_score(terms, title) < 0.20:
            continue
        seen.add(title.lower()[:80])
        pub_date = _format_pub_date((item.findtext("pubDate") or "").strip())
        source_el = item.find("source")
        source_name = (source_el.text or f"News-{gl}").strip() if source_el is not None else f"News-{gl}"
        if source_name.lower() in _BLOCKED_NEWS_SOURCES:
            continue
        snippets.append(
            NewsSnippet(
                tier=f"TIER 2 - REGIONAL PRESS ({gl})",
                source=source_name,
                published=pub_date,
                text=title,
            )
        )
        if len(snippets) >= max_items:
            break
    return snippets
