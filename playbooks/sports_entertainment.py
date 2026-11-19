"""Playbook operativo para la categoría sports_entertainment (~9% del torneo).

Combina mercados de predicción (Polymarket/Manifold), ESPN Public Scoreboard API
(récords y líneas/cuotas implícitas), prensa deportiva/entretenimiento curada y
calibración específica para la alta varianza de partidos únicos y premios.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import requests

from oracle_enrichment import fetch_polymarket_prior
from playbooks.odds_sources import OddsConsensus, fetch_the_odds_api_consensus
from playbooks.open_sources import (
    MarketMatch,
    NewsSnippet,
    _extract_focused_terms,
    _overlap_score,
    fetch_google_news_rss,
    fetch_manifold_prior,
)
from playbooks.politics_geo_law import (
    MAX_EVIDENCE_CHARS,
    PlaybookContext,
    _extract_primary_market_prior,
    estimate_days_remaining,
)

ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/scoreboard"
_ESPN_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}
_ESPN_TIMEOUT = 8.0

_LEAGUE_HINTS: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("nhl", "red wings", "panthers", "bruins", "rangers", "oilers", "maple leafs", "hockey"), "hockey", "nhl"),
    (("nba", "lakers", "celtics", "warriors", "nuggets", "knicks", "bucks"), "basketball", "nba"),
    (("nfl", "chiefs", "eagles", "49ers", "cowboys", "bills", "ravens", "super bowl"), "football", "nfl"),
    (("mlb", "yankees", "dodgers", "astros", "braves", "american league", "national league", "mvp"), "baseball", "mlb"),
    (("bundesliga", "bayern", "dortmund", "leverkusen"), "soccer", "ger.1"),
    (("premier league", "arsenal", "liverpool", "manchester", "chelsea"), "soccer", "eng.1"),
    (("uefa", "nations league", "champions league", "world cup"), "soccer", "uefa.nations"),
)


@dataclass(frozen=True)
class SportsSubtypeSpec:
    """Configuración de fuentes y sesgos por subtipo deportivo/entretenimiento."""

    subtype: str
    news_domains: tuple[str, ...]
    apply_status_quo_bias: bool
    default_shrink_alpha: float
    market_weight: float
    outside_view_hint: str


SPORTS_SUBTYPES: dict[str, SportsSubtypeSpec] = {
    "match_outcome": SportsSubtypeSpec(
        "match_outcome",
        ("espn.com", "bbc.com", "cbssports.com", "nhl.com", "uefa.com", "skysports.com"),
        False, 0.78, 0.60,
        "Single matches (especially NHL/soccer) have high goal variance: strong favorites rarely exceed 65-75%. Anchor on vig-free odds and avoid >85% confidence.",
    ),
    "tournament_advancement": SportsSubtypeSpec(
        "tournament_advancement",
        ("espn.com", "bbc.com", "iihf.com", "uefa.com", "reuters.com"),
        False, 0.80, 0.55,
        "Weigh historical seeding, FIFA/IIHF rankings, and bracket path; knockout rounds amplify upset risk.",
    ),
    "awards_records": SportsSubtypeSpec(
        "awards_records",
        ("espn.com", "mlb.com", "cbssports.com", "variety.com", "hollywoodreporter.com"),
        True, 0.80, 0.55,
        "For journalist-voted awards (MVP, Oscars), check straw polls and voter trackers. Unanimous selections have a low base rate.",
    ),
    "entertainment_media": SportsSubtypeSpec(
        "entertainment_media",
        ("variety.com", "hollywoodreporter.com", "boxofficemojo.com", "billboard.com", "bbc.com"),
        False, 0.82, 0.50,
        "Anchor on official box-office presales, midweek chart projections, and streaming rankings over social buzz.",
    ),
}

_AWARDS_TERMS = ("mvp", "unanimous", "award", "oscar", "grammy", "emmy", "ballon d'or", "cy young", "heisman")
_ENTERTAINMENT_TERMS = ("box office", "movie", "film", "album", "billboard", "spotify", "netflix", "rotten tomatoes", "grossing")
_TOURNAMENT_TERMS = ("semi-final", "semifinal", "quarter-final", "world championship", "playoffs", "qualify", "advance", "win the title")


def detect_sports_subtype(question_text: str) -> str:
    """Clasifica una pregunta de sports_entertainment en su subtipo operativo."""
    text = question_text.lower()
    if any(t in text for t in _AWARDS_TERMS):
        return "awards_records"
    if any(t in text for t in _ENTERTAINMENT_TERMS):
        return "entertainment_media"
    if any(t in text for t in _TOURNAMENT_TERMS):
        return "tournament_advancement"
    return "match_outcome"


def detect_espn_league(question_text: str) -> tuple[str, str] | None:
    """Detecta deporte y liga de ESPN si la pregunta corresponde a una liga cubierta."""
    text = question_text.lower()
    for keywords, sport, league in _LEAGUE_HINTS:
        if any(kw in text for kw in keywords):
            return sport, league
    return None


def fetch_espn_matchup_context(
    question_text: str,
    *,
    session: requests.Session | None = None,
) -> list[str]:
    """Consulta ESPN Scoreboard abierto para extraer partidos, récords y cuotas."""
    league_info = detect_espn_league(question_text)
    if league_info is None:
        return []

    sport, league = league_info
    terms = _extract_focused_terms(question_text, max_terms=6)
    client = session or requests.Session()
    try:
        resp = client.get(
            ESPN_SCOREBOARD_URL.format(sport=sport, league=league),
            headers=_ESPN_HEADERS,
            timeout=_ESPN_TIMEOUT,
        )
        if resp.status_code != 200:
            return []
        payload = resp.json()
    except (requests.RequestException, ValueError):
        return []

    events = payload.get("events") if isinstance(payload, dict) else None
    if not isinstance(events, list):
        return []

    lines: list[str] = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        name = str(ev.get("name") or "").strip()
        if not name or _overlap_score(terms, name) < 0.20:
            continue
        date_s = str(ev.get("date") or "")[:10]
        comps = ev.get("competitions") if isinstance(ev.get("competitions"), list) else []
        detail_parts = _extract_espn_comp_details(comps[0]) if comps and isinstance(comps[0], dict) else []
        suffix = f" | {', '.join(detail_parts)}" if detail_parts else ""
        lines.append(f"- [{date_s}] {name}{suffix}")
        if len(lines) >= 3:
            break
    return lines


def _extract_espn_comp_details(comp: dict) -> list[str]:
    parts: list[str] = []
    for team in comp.get("competitors") or []:
        if not isinstance(team, dict):
            continue
        abbr = str((team.get("team") or {}).get("abbreviation") or "")
        recs = team.get("records") or []
        rec = str(recs[0].get("summary") or "") if recs and isinstance(recs[0], dict) else ""
        if abbr:
            parts.append(f"{abbr} ({rec})" if rec else abbr)
    odds = comp.get("odds") or []
    if odds and isinstance(odds[0], dict):
        if odds[0].get("details"):
            parts.append(f"odds={str(odds[0]['details']).strip()}")
        if odds[0].get("overUnder") is not None:
            parts.append(f"O/U={odds[0]['overUnder']}")
    return parts


def _format_sports_evidence(
    *,
    subtype: str,
    days_remaining: int | None,
    odds_consensus: OddsConsensus | None,
    espn_lines: list[str],
    polymarket_block: str,
    manifold_matches: list[MarketMatch],
    news_items: list[NewsSnippet],
) -> str:
    sections: list[str] = [
        f"### Playbook Metadata\n- Domain: sports_entertainment ({subtype})\n"
        f"- Estimated days to resolution: {days_remaining if days_remaining is not None else 'unknown'}"
    ]
    if odds_consensus is not None:
        probs_str = " | ".join(
            f"{k}: {v:.1%}" for k, v in odds_consensus.outcome_probs.items()
        )
        sections.append(
            "### [TIER 1 - BOOKMAKER VIG-FREE CONSENSUS (THE ODDS API)]\n"
            f"- [{odds_consensus.commence_time}] {odds_consensus.event_name} "
            f"({odds_consensus.bookmaker_count} books): {probs_str}"
        )
    if espn_lines:
        sections.append("### [TIER 1 - ESPN SCOREBOARD & ODDS]\n" + "\n".join(espn_lines))
    if polymarket_block:
        sections.append(polymarket_block)
    if manifold_matches:
        m_lines = [
            f"- {m.title} (Manifold prob={m.probability:.1%}, vol={m.volume:,.0f})"
            if m.probability is not None
            else f"- {m.title} (Manifold multi-option, vol={m.volume:,.0f})"
            for m in manifold_matches
        ]
        sections.append("### [TIER 1 - PREDICTION MARKETS (MANIFOLD)]\n" + "\n".join(m_lines))
    if news_items:
        n_lines = [f"- [{n.published}] ({n.source}) {n.text}" for n in news_items]
        sections.append(f"### [{news_items[0].tier}]\n" + "\n".join(n_lines))

    combined = "\n\n".join(sections).strip()
    if len(combined) > MAX_EVIDENCE_CHARS:
        return combined[: MAX_EVIDENCE_CHARS - 20].rstrip() + "\n[...truncated]"
    return combined


def build_sports_entertainment_context(
    question_text: str,
    *,
    close_time: datetime | None = None,
    session: requests.Session | None = None,
    now: datetime | None = None,
) -> PlaybookContext:
    """Construye contexto completo (mercados + ESPN + prensa) para sports_entertainment."""
    subtype = detect_sports_subtype(question_text)
    spec = SPORTS_SUBTYPES[subtype]
    days_remaining = estimate_days_remaining(question_text, close_time=close_time, now=now)

    client = session or requests.Session()
    odds_consensus = (
        fetch_the_odds_api_consensus(
            question_text,
            days_remaining=days_remaining,
            session=client,
        )
        if subtype == "match_outcome"
        else None
    )
    espn_lines = fetch_espn_matchup_context(question_text, session=client)
    polymarket_block = fetch_polymarket_prior(question_text, session=client)
    manifold_matches = fetch_manifold_prior(question_text, session=client)
    news_items = fetch_google_news_rss(
        question_text,
        domains=spec.news_domains,
        session=client,
    )

    market_prior, market_source = _extract_primary_market_prior(
        polymarket_block, manifold_matches
    )
    if odds_consensus is not None and odds_consensus.target_team_prob is not None:
        market_prior = odds_consensus.target_team_prob
        market_source = f"TheOddsAPI ({odds_consensus.bookmaker_count} books vig-free)"

    evidence_text = _format_sports_evidence(
        subtype=subtype,
        days_remaining=days_remaining,
        odds_consensus=odds_consensus,
        espn_lines=espn_lines,
        polymarket_block=polymarket_block,
        manifold_matches=manifold_matches,
        news_items=news_items,
    )
    prompt_addendum = (
        "### Domain Forecasting Protocol (sports_entertainment)\n"
        f"- Subtype guidance ({spec.subtype}): {spec.outside_view_hint}\n"
        "- Calibration rule: respect inherent game variance and avoid >85% "
        "extremes on single matches unless the event is already decided.\n"
        "- Pre-mortem required: note injuries, goaltending/red-card variance, "
        "or voter dissent that could cause an upset."
    )

    return PlaybookContext(
        playbook="sports_entertainment",
        subtype=subtype,
        days_remaining=days_remaining,
        market_prior=market_prior,
        market_prior_source=market_source,
        apply_status_quo_bias=spec.apply_status_quo_bias,
        default_shrink_alpha=spec.default_shrink_alpha,
        market_weight=spec.market_weight,
        evidence_text=evidence_text,
        prompt_addendum=prompt_addendum,
    )
