"""Conectores de fuentes abiertas (0 tokens LLM, sin API key) para playbooks.

Todas las llamadas usan timeout estricto y fallan de forma silenciosa/tolerante
para no bloquear nunca una corrida del bot en torneo.
"""

from __future__ import annotations

from dataclasses import dataclass
from email.utils import parsedate_to_datetime
import re
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

import requests

from oracle_enrichment import extract_query_terms

MANIFOLD_SEARCH_URL = "https://api.manifold.markets/v0/search-markets"
GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"
BLUESKY_AUTHOR_FEED_URL = "https://public.api.bsky.app/xrpc/app.bsky.feed.getAuthorFeed"

HTTP_TIMEOUT_SECONDS = 8.0
DEFAULT_HEADERS = {
    "User-Agent": "WonkaForecaster/1.0 (Metaculus research bot; read-only)"
}
_BLOCKED_NEWS_SOURCES = frozenset({
    "facebook", "instagram", "tiktok", "pinterest", "reddit", "x", "twitter",
})


@dataclass(frozen=True)
class MarketMatch:
    """Referencia estructurada a un mercado de predicción externo."""

    source: str
    title: str
    probability: float | None
    volume: float
    overlap_score: float


@dataclass(frozen=True)
class NewsSnippet:
    """Titular normalizado proveniente de prensa curada o cuenta OSINT."""

    tier: str
    source: str
    published: str
    text: str


_GENERIC_VERBS = frozenset({
    "publicly", "stated", "cause", "report", "reported", "launched", "lead",
    "leading", "pending", "against", "conclude", "proceedings", "largest",
    "difference", "among", "second", "stage", "night", "first", "choice",
    "total", "active", "least", "reach", "exceed", "close", "closing",
})


def _extract_focused_terms(question_text: str, max_terms: int = 6) -> list[str]:
    """Prioriza entidades capitalizadas y siglas sobre verbos genéricos."""
    base_terms = extract_query_terms(question_text, max_terms=16)
    # Detectar palabras con mayúscula interna o inicial (fuera del primer token)
    raw_words = re.findall(r"[A-Za-z0-9]+", question_text)
    entities = {
        w.lower()
        for idx, w in enumerate(raw_words)
        if idx > 0 and (w[0].isupper() or w.isupper()) and len(w) >= 2
    }
    prioritized = [t for t in base_terms if t in entities and t not in _GENERIC_VERBS]
    secondary = [t for t in base_terms if t not in prioritized and t not in _GENERIC_VERBS]
    return (prioritized + secondary)[:max_terms]


def _overlap_score(query_terms: list[str], candidate_text: str) -> float:
    if not query_terms:
        return 0.0
    tokens = set(re.findall(r"[a-zA-Z0-9]+", candidate_text.lower()))
    hits = sum(1 for t in query_terms if t in tokens)
    return hits / len(query_terms)


def _format_pub_date(raw_date: str) -> str:
    if not raw_date:
        return "unknown-date"
    try:
        dt = parsedate_to_datetime(raw_date)
        return dt.strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        return raw_date[:10]


def fetch_manifold_prior(
    question_text: str,
    *,
    session: requests.Session | None = None,
    min_overlap: float = 0.45,
    min_traders: int = 10,
) -> list[MarketMatch]:
    """Busca mercados equivalentes en Manifold Markets (API abierta sin key)."""
    terms = extract_query_terms(question_text, max_terms=6)
    if len(terms) < 2:
        return []

    client = session or requests.Session()
    try:
        resp = client.get(
            MANIFOLD_SEARCH_URL,
            params={"term": " ".join(terms[:4]), "limit": 8},
            headers=DEFAULT_HEADERS,
            timeout=HTTP_TIMEOUT_SECONDS,
        )
        if resp.status_code != 200:
            return []
        payload = resp.json()
    except (requests.RequestException, ValueError):
        return []

    if not isinstance(payload, list):
        return []

    matches: list[MarketMatch] = []
    for item in payload:
        if not isinstance(item, dict) or item.get("isResolved"):
            continue
        traders = int(item.get("uniqueBettorCount") or 0)
        if traders < min_traders:
            continue
        title = str(item.get("question") or "").strip()
        score = _overlap_score(terms, title)
        if score < min_overlap:
            continue
        prob_raw = item.get("probability")
        prob = float(prob_raw) if isinstance(prob_raw, (int, float)) else None
        if prob is not None and not (0.0 <= prob <= 1.0):
            prob = None
        volume = float(item.get("volume") or 0.0)
        matches.append(
            MarketMatch(
                source="Manifold",
                title=title,
                probability=prob,
                volume=volume,
                overlap_score=score,
            )
        )

    matches.sort(key=lambda m: (m.overlap_score, m.volume), reverse=True)
    return matches[:2]


def fetch_google_news_rss(
    question_text: str,
    *,
    domains: tuple[str, ...] = (),
    session: requests.Session | None = None,
    max_items: int = 6,
) -> list[NewsSnippet]:
    """Consulta Google News RSS apuntando a dominios curados con fallback abierto."""
    terms = _extract_focused_terms(question_text, max_terms=6)
    if len(terms) < 2:
        return []

    base_query = " ".join(terms[:5])
    queries: list[tuple[str, str]] = []
    if domains:
        site_filter = " OR ".join(f"site:{d}" for d in domains)
        queries.append((f"{base_query} ({site_filter}) when:60d", "TIER 2 - CURATED PRESS"))
    queries.append((f"{base_query} when:60d", "TIER 2 - GENERAL PRESS"))

    client = session or requests.Session()
    seen_titles: set[str] = set()
    snippets: list[NewsSnippet] = []

    for query, tier_label in queries:
        if len(snippets) >= max_items:
            break
        url = f"{GOOGLE_NEWS_RSS_URL}?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
        try:
            resp = client.get(url, headers=DEFAULT_HEADERS, timeout=HTTP_TIMEOUT_SECONDS)
            if resp.status_code != 200:
                continue
            root = ET.fromstring(resp.text)
        except (requests.RequestException, ET.ParseError):
            continue

        for item in root.findall(".//item"):
            title = (item.findtext("title") or "").strip()
            pub_date = _format_pub_date((item.findtext("pubDate") or "").strip())
            source_el = item.find("source")
            source_name = (source_el.text or "News").strip() if source_el is not None else "News"
            if source_name.lower() in _BLOCKED_NEWS_SOURCES:
                continue

            norm_key = title.lower()[:80]
            if not title or norm_key in seen_titles:
                continue
            if _overlap_score(terms, title) < 0.20:
                continue
            seen_titles.add(norm_key)
            snippets.append(
                NewsSnippet(
                    tier=tier_label,
                    source=source_name,
                    published=pub_date,
                    text=title,
                )
            )
            if len(snippets) >= max_items:
                break

        # Si los dominios curados ya trajeron al menos 3 titulares, no agregamos ruido abierto
        if len(snippets) >= 3:
            break

    return snippets


def fetch_bluesky_watchlist(
    question_text: str,
    *,
    handles: tuple[str, ...],
    session: requests.Session | None = None,
    max_items: int = 4,
) -> list[NewsSnippet]:
    """Lee posts recientes de cuentas de Bluesky apuntadas y filtra por relevancia."""
    if not handles:
        return []
    terms = extract_query_terms(question_text, max_terms=6)
    if len(terms) < 2:
        return []

    client = session or requests.Session()
    hits: list[tuple[float, NewsSnippet]] = []

    for handle in handles[:3]:
        try:
            resp = client.get(
                BLUESKY_AUTHOR_FEED_URL,
                params={"actor": handle, "limit": 20},
                headers=DEFAULT_HEADERS,
                timeout=HTTP_TIMEOUT_SECONDS,
            )
            if resp.status_code != 200:
                continue
            payload = resp.json()
        except (requests.RequestException, ValueError):
            continue

        feed = payload.get("feed") if isinstance(payload, dict) else None
        if not isinstance(feed, list):
            continue

        for entry in feed:
            post = entry.get("post") if isinstance(entry, dict) else None
            record = post.get("record") if isinstance(post, dict) else None
            if not isinstance(record, dict):
                continue
            text = " ".join(str(record.get("text") or "").split())
            if not text:
                continue
            score = _overlap_score(terms, text)
            if score < 0.40:
                continue
            created = str(record.get("createdAt") or "")[:10] or "unknown-date"
            hits.append(
                (
                    score,
                    NewsSnippet(
                        tier="TIER 3 - BLUESKY WATCHLIST (VERIFY)",
                        source=f"@{handle}",
                        published=created,
                        text=text[:220],
                    ),
                )
            )

    hits.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in hits[:max_items]]
