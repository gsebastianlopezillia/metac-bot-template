"""Conectores financieros y macroeconómicos (0 tokens LLM) para economy_business.

Fuentes soportadas:
- Yahoo Finance Chart API pública (sin key): precios, retornos y volatilidad histórica.
- FRED CSV público (sin key): observaciones oficiales de series macro (ej. FYONET, CPIAUCSL).
- Alpaca News API (opcional, si ALPACA_API_KEY/ALPACA_SECRET_KEY están en .env): noticias
  de Benzinga filtradas por ticker exacto.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
import os
import re

import requests

from playbooks.open_sources import DEFAULT_HEADERS, HTTP_TIMEOUT_SECONDS, NewsSnippet

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
ALPACA_NEWS_URL = "https://data.alpaca.markets/v1beta1/news"

_COMPANY_TO_TICKER: dict[str, str] = {
    "amazon": "AMZN",
    "alphabet": "GOOGL",
    "google": "GOOGL",
    "applied materials": "AMAT",
    "apple": "AAPL",
    "microsoft": "MSFT",
    "nvidia": "NVDA",
    "tesla": "TSLA",
    "meta": "META",
    "okta": "OKTA",
    "berkshire hathaway": "BRK-B",
    "netflix": "NFLX",
    "intel": "INTC",
    "amd": "AMD",
}

_KNOWN_FRED_SERIES = frozenset({
    "FYONET", "CPIAUCSL", "CPILFESL", "UNRATE", "GDP", "GDPC1",
    "FEDFUNDS", "DGS10", "DGS2", "PCEPI", "PCEPILFE", "PAYEMS",
    "INDPRO", "MORTGAGE30US", "DFF", "T10Y2Y", "ICSA", "BAMLH0A0HYM2",
})

_NON_TICKER_ACRONYMS = frozenset({
    "SEC", "ETF", "GDP", "IEA", "USA", "USD", "EUR", "CBO", "FED", "OPEC",
    "AST", "FRED", "API", "UTC", "CEO", "CFO", "IPO", "FOMC", "CPI", "PPI",
    "YOY", "QOQ", "FY", "Q1", "Q2", "Q3", "Q4", "EST", "EDT", "PDT", "MB",
})

_PAREN_TICKER_RE = re.compile(r"\(([A-Z]{2,5})\)")
_ETF_TICKER_RE = re.compile(r"\b(?:ETF|ticker|symbol)\s+([A-Z]{2,5})\b", re.IGNORECASE)
_FRED_EXPLICIT_RE = re.compile(r"\bFRED(?:\s+series)?\s+[`'\"]?([A-Z0-9_]{3,16})[`'\"]?", re.IGNORECASE)


@dataclass(frozen=True)
class EquityStats:
    """Resumen estadístico de un ticker obtenido desde Yahoo Finance."""

    ticker: str
    last_date: str
    last_close: float
    ret_14d_pct: float | None
    ret_30d_pct: float | None
    daily_vol_pct: float | None
    expected_10d_move_pct: float | None


@dataclass(frozen=True)
class FredSeriesSummary:
    """Últimas observaciones oficiales de una serie de FRED."""

    series_id: str
    observations: tuple[tuple[str, float], ...]
    yoy_change_pct: float | None


def extract_tickers(question_text: str, max_tickers: int = 3) -> list[str]:
    """Extrae tickers bursátiles confiables de la pregunta."""
    found: list[str] = []

    for match in _PAREN_TICKER_RE.findall(question_text):
        if match not in _NON_TICKER_ACRONYMS and match not in found:
            found.append(match)

    for match in _ETF_TICKER_RE.findall(question_text):
        sym = match.upper()
        if sym not in _NON_TICKER_ACRONYMS and sym not in found:
            found.append(sym)

    lower = question_text.lower()
    for name, sym in _COMPANY_TO_TICKER.items():
        if name in lower and sym not in found:
            found.append(sym)

    return found[:max_tickers]


def extract_fred_series_ids(text: str, max_series: int = 2) -> list[str]:
    """Detecta códigos de series FRED mencionados explícitamente en texto o criterios."""
    found: list[str] = []
    for match in _FRED_EXPLICIT_RE.findall(text):
        code = match.upper()
        if code not in found:
            found.append(code)

    tokens = set(re.findall(r"\b[A-Z0-9_]{3,16}\b", text))
    for known in sorted(_KNOWN_FRED_SERIES):
        if known in tokens and known not in found:
            found.append(known)

    return found[:max_series]


def fetch_yahoo_stats(
    ticker: str,
    *,
    session: requests.Session | None = None,
) -> EquityStats | None:
    """Obtiene precio actual, retornos y volatilidad histórica diaria desde Yahoo Finance."""
    client = session or requests.Session()
    try:
        resp = client.get(
            YAHOO_CHART_URL.format(ticker=ticker),
            params={"range": "3mo", "interval": "1d"},
            headers=DEFAULT_HEADERS,
            timeout=HTTP_TIMEOUT_SECONDS,
        )
        if resp.status_code != 200:
            return None
        payload = resp.json()
    except (requests.RequestException, ValueError):
        return None

    results = (payload.get("chart") or {}).get("result") if isinstance(payload, dict) else None
    if not isinstance(results, list) or not results:
        return None

    item = results[0]
    timestamps = item.get("timestamp") or []
    quotes = ((item.get("indicators") or {}).get("quote") or [{}])[0]
    raw_closes = quotes.get("close") or []

    series: list[tuple[int, float]] = [
        (int(ts), float(c))
        for ts, c in zip(timestamps, raw_closes)
        if ts is not None and isinstance(c, (int, float)) and c > 0
    ]
    if len(series) < 5:
        return None

    last_ts, last_close = series[-1]
    last_date = datetime.fromtimestamp(last_ts, tz=timezone.utc).strftime("%Y-%m-%d")

    ret_14d = (
        ((last_close / series[-11][1]) - 1.0) * 100.0
        if len(series) >= 11
        else None
    )
    ret_30d = (
        ((last_close / series[-22][1]) - 1.0) * 100.0
        if len(series) >= 22
        else None
    )

    log_returns = [
        math.log(series[i][1] / series[i - 1][1])
        for i in range(1, len(series))
    ]
    mean_r = sum(log_returns) / len(log_returns)
    variance = sum((r - mean_r) ** 2 for r in log_returns) / max(1, len(log_returns) - 1)
    daily_vol = math.sqrt(variance) * 100.0
    expected_10d = daily_vol * math.sqrt(10.0)

    return EquityStats(
        ticker=ticker,
        last_date=last_date,
        last_close=round(last_close, 2),
        ret_14d_pct=round(ret_14d, 2) if ret_14d is not None else None,
        ret_30d_pct=round(ret_30d, 2) if ret_30d is not None else None,
        daily_vol_pct=round(daily_vol, 2),
        expected_10d_move_pct=round(expected_10d, 2),
    )


def fetch_fred_series(
    series_id: str,
    *,
    session: requests.Session | None = None,
    tail_points: int = 5,
) -> FredSeriesSummary | None:
    """Descarga el CSV público de FRED (sin API key) y extrae las últimas observaciones."""
    client = session or requests.Session()
    try:
        resp = client.get(
            FRED_CSV_URL,
            params={"id": series_id},
            headers=DEFAULT_HEADERS,
            timeout=HTTP_TIMEOUT_SECONDS,
        )
        if resp.status_code != 200:
            return None
    except requests.RequestException:
        return None

    rows: list[tuple[str, float]] = []
    for line in resp.text.splitlines()[1:]:
        parts = line.strip().split(",")
        if len(parts) != 2 or not parts[1] or parts[1] == ".":
            continue
        try:
            rows.append((parts[0], float(parts[1])))
        except ValueError:
            continue

    if not rows:
        return None

    recent = tuple(rows[-tail_points:])
    yoy_pct = None
    if len(recent) >= 2 and recent[-2][1] != 0:
        yoy_pct = round(((recent[-1][1] / recent[-2][1]) - 1.0) * 100.0, 2)

    return FredSeriesSummary(
        series_id=series_id,
        observations=recent,
        yoy_change_pct=yoy_pct,
    )


def fetch_alpaca_news(
    tickers: list[str],
    *,
    session: requests.Session | None = None,
    limit: int = 4,
) -> list[NewsSnippet]:
    """Consulta Alpaca/Benzinga News por ticker si las credenciales están en el entorno."""
    if not tickers:
        return []
    api_key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
    api_secret = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
    if not api_key or not api_secret:
        return []

    client = session or requests.Session()
    try:
        resp = client.get(
            ALPACA_NEWS_URL,
            params={"symbols": ",".join(tickers), "limit": limit},
            headers={
                "APCA-API-KEY-ID": api_key,
                "APCA-API-SECRET-KEY": api_secret,
            },
            timeout=HTTP_TIMEOUT_SECONDS,
        )
        if resp.status_code != 200:
            return []
        payload = resp.json()
    except (requests.RequestException, ValueError):
        return []

    articles = payload.get("news") if isinstance(payload, dict) else None
    if not isinstance(articles, list):
        return []

    snippets: list[NewsSnippet] = []
    for art in articles[:limit]:
        if not isinstance(art, dict):
            continue
        headline = " ".join(str(art.get("headline") or "").split())
        if not headline:
            continue
        created = str(art.get("created_at") or "")[:10] or "unknown-date"
        source = str(art.get("source") or "Benzinga/Alpaca").strip()
        snippets.append(
            NewsSnippet(
                tier="TIER 2 - ALPACA/BENZINGA NEWS",
                source=source,
                published=created,
                text=headline[:220],
            )
        )
    return snippets
