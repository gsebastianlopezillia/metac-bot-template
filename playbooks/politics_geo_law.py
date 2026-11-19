"""Playbook operativo para la categoría politics_geo_law (~42% del torneo).

Integra fuentes abiertas curadas (0 tokens LLM), cálculo de horizonte temporal
y reglas de razonamiento estructurado (Outside View, Status Quo y Pre-Mortem).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re

import requests

from oracle_enrichment import fetch_polymarket_prior
from playbooks.open_sources import (
    MarketMatch,
    NewsSnippet,
    fetch_bluesky_watchlist,
    fetch_google_news_rss,
    fetch_manifold_prior,
)
from playbooks.watchlists import (
    POLITICS_WATCHLISTS,
    SubtypeWatchlist,
    detect_politics_subtype,
)

MAX_EVIDENCE_CHARS = 3200
_POLYMARKET_PROB_RE = re.compile(r"YES:\s*(\d+(?:\.\d+)?)%")
_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}
_DATE_MDY_RE = re.compile(
    r"\b(" + "|".join(_MONTH_MAP.keys()) + r")\s+(\d{1,2}),?\s+(20\d{2})\b",
    re.IGNORECASE,
)
_DATE_DMY_RE = re.compile(
    r"\b(\d{1,2})\s+(" + "|".join(_MONTH_MAP.keys()) + r")\s+(20\d{2})\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PlaybookContext:
    """Paquete estructurado que devuelve el playbook antes de llamar al LLM."""

    playbook: str
    subtype: str
    days_remaining: int | None
    market_prior: float | None
    market_prior_source: str | None
    apply_status_quo_bias: bool
    default_shrink_alpha: float
    market_weight: float
    evidence_text: str
    prompt_addendum: str


def estimate_days_remaining(
    question_text: str,
    close_time: datetime | None = None,
    now: datetime | None = None,
) -> int | None:
    """Estima días hasta la resolución usando close_time o fechas en el texto."""
    ref_now = now or datetime.now(timezone.utc)
    if close_time is not None:
        target = close_time if close_time.tzinfo else close_time.replace(tzinfo=timezone.utc)
        return max(0, (target - ref_now).days)

    candidates: list[datetime] = []
    for month_name, day_s, year_s in _DATE_MDY_RE.findall(question_text):
        try:
            candidates.append(
                datetime(int(year_s), _MONTH_MAP[month_name.lower()], int(day_s), tzinfo=timezone.utc)
            )
        except ValueError:
            continue
    for day_s, month_name, year_s in _DATE_DMY_RE.findall(question_text):
        try:
            candidates.append(
                datetime(int(year_s), _MONTH_MAP[month_name.lower()], int(day_s), tzinfo=timezone.utc)
            )
        except ValueError:
            continue

    if not candidates:
        return None
    latest = max(candidates)
    return max(0, (latest - ref_now).days)


def _extract_primary_market_prior(
    polymarket_block: str,
    manifold_matches: list[MarketMatch],
) -> tuple[float | None, str | None]:
    """Obtiene una probabilidad ancla [0..1] priorizando Polymarket y luego Manifold."""
    if polymarket_block:
        match = _POLYMARKET_PROB_RE.search(polymarket_block)
        if match:
            val = float(match.group(1)) / 100.0
            if 0.01 <= val <= 0.99:
                return val, "Polymarket"

    for item in manifold_matches:
        if item.probability is not None and 0.01 <= item.probability <= 0.99:
            return item.probability, "Manifold"

    return None, None


def _format_evidence_block(
    *,
    subtype: str,
    days_remaining: int | None,
    polymarket_block: str,
    manifold_matches: list[MarketMatch],
    news_items: list[NewsSnippet],
    bluesky_items: list[NewsSnippet],
) -> str:
    sections: list[str] = [
        f"### Playbook Metadata\n- Domain: politics_geo_law ({subtype})\n"
        f"- Estimated days to resolution: {days_remaining if days_remaining is not None else 'unknown'}"
    ]

    tier1_lines: list[str] = []
    if polymarket_block:
        tier1_lines.append(polymarket_block)
    if manifold_matches:
        m_lines = [
            f"- {m.title} (Manifold prob={m.probability:.1%}, vol={m.volume:,.0f})"
            if m.probability is not None
            else f"- {m.title} (Manifold multi-option, vol={m.volume:,.0f})"
            for m in manifold_matches
        ]
        tier1_lines.append("### [TIER 1 - PREDICTION MARKETS (MANIFOLD)]\n" + "\n".join(m_lines))
    if tier1_lines:
        sections.append("\n\n".join(tier1_lines))

    if news_items:
        n_lines = [
            f"- [{n.published}] ({n.source}) {n.text}"
            for n in news_items
        ]
        sections.append(f"### [{news_items[0].tier}]\n" + "\n".join(n_lines))

    if bluesky_items:
        b_lines = [
            f"- [{b.published}] ({b.source}) {b.text}"
            for b in bluesky_items
        ]
        sections.append(
            "### [TIER 3 - BLUESKY WATCHLIST (EARLY SIGNAL, VERIFY AGAINST TIER 1/2)]\n"
            + "\n".join(b_lines)
        )

    combined = "\n\n".join(sections).strip()
    if len(combined) > MAX_EVIDENCE_CHARS:
        return combined[: MAX_EVIDENCE_CHARS - 20].rstrip() + "\n[...truncated]"
    return combined


def _build_prompt_addendum(watchlist: SubtypeWatchlist, days_remaining: int | None) -> str:
    horizon_note = (
        f"Short horizon ({days_remaining} days left): apply strong status-quo inertia "
        "unless concrete procedural steps are already underway."
        if watchlist.apply_status_quo_bias and days_remaining is not None and days_remaining <= 14
        else "Weigh institutional friction and historical base rates before updating on headlines."
    )
    return (
        "### Domain Forecasting Protocol (politics_geo_law)\n"
        f"- Subtype guidance ({watchlist.subtype}): {watchlist.outside_view_hint}\n"
        f"- Horizon rule: {horizon_note}\n"
        "- Source hierarchy: trust TIER 1 (markets/official data) > TIER 2 (curated press) "
        "> TIER 3 (social/OSINT). Discount speculative headlines without docket/official confirmation.\n"
        "- Pre-mortem required: before finalizing numbers, state the single most plausible "
        "bureaucratic, legal, or statistical surprise that could make your forecast wrong."
    )


def build_politics_geo_law_context(
    question_text: str,
    *,
    official_categories: list[str] | tuple[str, ...] = (),
    close_time: datetime | None = None,
    session: requests.Session | None = None,
    now: datetime | None = None,
) -> PlaybookContext:
    """Construye contexto completo (fuentes curadas + sesgos) para politics_geo_law."""
    subtype = detect_politics_subtype(question_text, official_categories)
    watchlist = POLITICS_WATCHLISTS[subtype]
    days_remaining = estimate_days_remaining(question_text, close_time=close_time, now=now)

    client = session or requests.Session()
    polymarket_block = fetch_polymarket_prior(question_text, session=client)
    manifold_matches = fetch_manifold_prior(question_text, session=client)
    news_items = fetch_google_news_rss(
        question_text,
        domains=watchlist.news_domains,
        session=client,
    )
    bluesky_items = fetch_bluesky_watchlist(
        question_text,
        handles=watchlist.bluesky_handles,
        session=client,
    )

    market_prior, market_source = _extract_primary_market_prior(
        polymarket_block, manifold_matches
    )
    evidence_text = _format_evidence_block(
        subtype=subtype,
        days_remaining=days_remaining,
        polymarket_block=polymarket_block,
        manifold_matches=manifold_matches,
        news_items=news_items,
        bluesky_items=bluesky_items,
    )
    prompt_addendum = _build_prompt_addendum(watchlist, days_remaining)

    return PlaybookContext(
        playbook="politics_geo_law",
        subtype=subtype,
        days_remaining=days_remaining,
        market_prior=market_prior,
        market_prior_source=market_source,
        apply_status_quo_bias=watchlist.apply_status_quo_bias,
        default_shrink_alpha=watchlist.default_shrink_alpha,
        market_weight=watchlist.market_weight,
        evidence_text=evidence_text,
        prompt_addendum=prompt_addendum,
    )
