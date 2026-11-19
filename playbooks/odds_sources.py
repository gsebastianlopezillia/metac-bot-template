"""Conector de cuotas de apuestas (The Odds API) con matemática vig-free y control de cuota.

Protecciones estrictas para el plan gratuito de 500 créditos/mes:
1. Consulta 1 sola región ('us') y 1 mercado ('h2h') = exactamente 1 crédito por liga.
2. Caché en disco por liga con TTL de 6 horas (varias preguntas de NHL/NBA = 1 crédito).
3. Solo se invoca en partidos con horizonte <= 14 días (o desconocido).
4. Si el header x-requests-remaining baja de 25, entra en modo reserva automático.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import statistics
import time
from typing import Any

import requests

from playbooks.open_sources import HTTP_TIMEOUT_SECONDS, _extract_focused_terms, _overlap_score

ODDS_API_EVENTS_URL = "https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
ODDS_CACHE_DIR = Path("benchmarks/cache/odds")
ODDS_CACHE_TTL_SECONDS = 6 * 3600
MIN_CREDITS_RESERVE = 25

_SPORT_KEY_HINTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("nhl", "red wings", "panthers", "bruins", "rangers", "oilers", "maple leafs"), "icehockey_nhl"),
    (("nba", "lakers", "celtics", "warriors", "nuggets", "knicks", "bucks"), "basketball_nba"),
    (("nfl", "chiefs", "eagles", "49ers", "cowboys", "bills", "ravens", "super bowl"), "americanfootball_nfl"),
    (("mlb", "yankees", "dodgers", "astros", "braves"), "baseball_mlb"),
    (("bundesliga", "bayern", "dortmund", "leverkusen"), "soccer_germany_bundesliga"),
    (("premier league", "arsenal", "liverpool", "manchester", "chelsea"), "soccer_epl"),
    (("uefa nations league", "nations league"), "soccer_uefa_nations_league"),
    (("champions league",), "soccer_uefa_champs_league"),
)


@dataclass(frozen=True)
class OddsConsensus:
    """Consenso vig-free de casas de apuestas para un partido."""

    sport_key: str
    event_name: str
    commence_time: str
    bookmaker_count: int
    outcome_probs: dict[str, float]
    target_team_prob: float | None


def american_odds_to_implied_prob(american: float) -> float:
    """Convierte cuota americana (-150, +130) a probabilidad implícita bruta."""
    if american < 0:
        return (-american) / (-american + 100.0)
    if american > 0:
        return 100.0 / (american + 100.0)
    raise ValueError("La cuota americana no puede ser 0")


def decimal_odds_to_implied_prob(decimal_odds: float) -> float:
    """Convierte cuota decimal (ej. 1.65, 2.40) a probabilidad implícita bruta."""
    if decimal_odds <= 1.0:
        raise ValueError("La cuota decimal debe ser > 1.0")
    return 1.0 / decimal_odds


def remove_vig(raw_probs: dict[str, float]) -> dict[str, float]:
    """Elimina el margen de la casa (vig/overround) normalizando a suma 1.0."""
    total = sum(v for v in raw_probs.values() if v > 0)
    if total <= 0:
        return {}
    return {k: round(v / total, 4) for k, v in raw_probs.items() if v > 0}


def detect_odds_api_sport_key(question_text: str) -> str | None:
    """Mapea la pregunta deportiva a su sport_key oficial de The Odds API."""
    text = question_text.lower()
    for keywords, sport_key in _SPORT_KEY_HINTS:
        if any(kw in text for kw in keywords):
            return sport_key
    return None


def _compute_event_vig_free_consensus(event: dict[str, Any]) -> tuple[dict[str, float], int]:
    """Calcula la mediana vig-free entre todos los bookmakers del evento."""
    per_outcome: dict[str, list[float]] = {}
    book_count = 0

    for book in event.get("bookmakers") or []:
        if not isinstance(book, dict):
            continue
        markets = book.get("markets") or []
        h2h = next((m for m in markets if isinstance(m, dict) and m.get("key") == "h2h"), None)
        if not h2h:
            continue
        raw_probs: dict[str, float] = {}
        for out in h2h.get("outcomes") or []:
            if not isinstance(out, dict):
                continue
            name = str(out.get("name") or "").strip()
            price = out.get("price")
            if name and isinstance(price, (int, float)) and price > 1.0:
                raw_probs[name] = decimal_odds_to_implied_prob(float(price))
        clean = remove_vig(raw_probs)
        if len(clean) >= 2:
            book_count += 1
            for name, prob in clean.items():
                per_outcome.setdefault(name, []).append(prob)

    if not per_outcome:
        return {}, 0

    medians = {name: statistics.median(vals) for name, vals in per_outcome.items()}
    return remove_vig(medians), book_count


def _infer_target_team_prob(question_text: str, outcome_probs: dict[str, float]) -> float | None:
    """En preguntas 'Will <Team A> beat <Team B>?', identifica la prob del sujeto."""
    lower = question_text.lower()
    if " beat " not in lower and " win " not in lower:
        return None
    subject_part = lower.split(" beat ")[0] if " beat " in lower else lower.split(" win ")[0]
    best_prob: float | None = None
    best_overlap = 0.0
    subject_terms = _extract_focused_terms(subject_part, max_terms=4)
    for team_name, prob in outcome_probs.items():
        if team_name.lower() == "draw":
            continue
        score = _overlap_score(subject_terms, team_name)
        if score > best_overlap:
            best_overlap = score
            best_prob = prob
    return best_prob if best_overlap >= 0.40 else None


def _load_or_fetch_league_events(
    sport_key: str,
    *,
    api_key: str,
    session: requests.Session | None,
    cache_dir: Path,
) -> list[dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    quota_file = cache_dir / "_quota_state.json"
    if quota_file.exists():
        try:
            q_state = json.loads(quota_file.read_text(encoding="utf-8"))
            if int(q_state.get("remaining", 999)) < MIN_CREDITS_RESERVE:
                return []
        except (ValueError, OSError):
            pass

    cache_file = cache_dir / f"odds_{sport_key}.json"
    if cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < ODDS_CACHE_TTL_SECONDS:
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                if isinstance(cached, list):
                    return cached
            except (ValueError, OSError):
                pass

    client = session or requests.Session()
    try:
        resp = client.get(
            ODDS_API_EVENTS_URL.format(sport_key=sport_key),
            params={
                "apiKey": api_key,
                "regions": "us",
                "markets": "h2h",
                "oddsFormat": "decimal",
            },
            timeout=HTTP_TIMEOUT_SECONDS,
        )
        rem = resp.headers.get("x-requests-remaining") if hasattr(resp, "headers") else None
        if rem is not None:
            try:
                quota_file.write_text(
                    json.dumps({"remaining": int(float(rem)), "updated_at": int(time.time())}),
                    encoding="utf-8",
                )
            except (ValueError, OSError):
                pass
        if resp.status_code != 200:
            return []
        events = resp.json()
    except (requests.RequestException, ValueError):
        return []

    if isinstance(events, list):
        try:
            cache_file.write_text(json.dumps(events), encoding="utf-8")
        except OSError:
            pass
        return events
    return []


def fetch_the_odds_api_consensus(
    question_text: str,
    *,
    days_remaining: int | None = None,
    session: requests.Session | None = None,
    cache_dir: Path = ODDS_CACHE_DIR,
) -> OddsConsensus | None:
    """Devuelve consenso vig-free de The Odds API cuidando el límite de 500 créditos/mes."""
    api_key = os.getenv("THE_ODDS_API_KEY")
    if not api_key:
        return None
    if days_remaining is not None and days_remaining > 14:
        return None

    sport_key = detect_odds_api_sport_key(question_text)
    if sport_key is None:
        return None

    events = _load_or_fetch_league_events(
        sport_key, api_key=api_key, session=session, cache_dir=cache_dir
    )
    if not events:
        return None

    terms = _extract_focused_terms(question_text, max_terms=6)
    best_event: dict[str, Any] | None = None
    best_score = 0.0
    for ev in events:
        if not isinstance(ev, dict):
            continue
        title = f"{ev.get('home_team', '')} vs {ev.get('away_team', '')}"
        score = _overlap_score(terms, title)
        if score > best_score:
            best_score = score
            best_event = ev

    if best_event is None or best_score < 0.30:
        return None

    outcome_probs, book_count = _compute_event_vig_free_consensus(best_event)
    if not outcome_probs:
        return None

    event_name = f"{best_event.get('home_team', '')} vs {best_event.get('away_team', '')}".strip()
    target_prob = _infer_target_team_prob(question_text, outcome_probs)
    return OddsConsensus(
        sport_key=sport_key,
        event_name=event_name,
        commence_time=str(best_event.get("commence_time") or "")[:10],
        bookmaker_count=book_count,
        outcome_probs=outcome_probs,
        target_team_prob=target_prob,
    )
