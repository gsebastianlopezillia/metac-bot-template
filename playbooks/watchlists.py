"""Listas blancas (watchlists) curadas por subtipo para politics_geo_law.

Permite "apuntar" medios, portales oficiales y cuentas de Bluesky específicas
sin tocar la lógica del bot. Cada fuente se consulta a 0 tokens de LLM.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SubtypeWatchlist:
    """Configuración curada de fuentes y sesgos para un subtipo político/legal."""

    subtype: str
    news_domains: tuple[str, ...]
    bluesky_handles: tuple[str, ...]
    apply_status_quo_bias: bool
    default_shrink_alpha: float
    market_weight: float
    outside_view_hint: str


POLITICS_WATCHLISTS: dict[str, SubtypeWatchlist] = {
    "elections": SubtypeWatchlist(
        subtype="elections",
        news_domains=(
            "apnews.com",
            "reuters.com",
            "politico.com",
            "ballotpedia.org",
            "bbc.com",
        ),
        bluesky_handles=(
            "apnews.com",
            "politico.com",
        ),
        apply_status_quo_bias=False,
        default_shrink_alpha=0.84,
        market_weight=0.55,
        outside_view_hint=(
            "Prioritize verified vote tallies, official electoral commissions, "
            "and high-volume prediction markets over partisan punditry. For "
            "numeric margins or max-of-N swing states, account for correlated "
            "polling errors (typically 2-4 percentage points)."
        ),
    ),
    "conflict_geo": SubtypeWatchlist(
        subtype="conflict_geo",
        news_domains=(
            "reuters.com",
            "apnews.com",
            "bbc.com",
            "kyivindependent.com",
            "understandingwar.org",
        ),
        bluesky_handles=(
            "reuters.com",
            "kyivindependent.com",
        ),
        apply_status_quo_bias=True,
        default_shrink_alpha=0.78,
        market_weight=0.45,
        outside_view_hint=(
            "Discount dramatic headlines and rhetoric; anchor on recent 7-14 "
            "day empirical cadence (e.g. nightly counts, confirmed operational "
            "tempo) and use wide fat-tailed intervals for numeric counts."
        ),
    ),
    "law_courts": SubtypeWatchlist(
        subtype="law_courts",
        news_domains=(
            "reuters.com",
            "apnews.com",
            "scotusblog.com",
            "courtlistener.com",
            "politico.com",
        ),
        bluesky_handles=(
            "apnews.com",
            "reuters.com",
        ),
        apply_status_quo_bias=True,
        default_shrink_alpha=0.80,
        market_weight=0.50,
        outside_view_hint=(
            "Legal proceedings move slowly and leave a formal docket trail. "
            "Verify whether indictments, motions, or hearings are already "
            "scheduled before assigning high probability to short-horizon changes."
        ),
    ),
    "parliament_bureaucracy": SubtypeWatchlist(
        subtype="parliament_bureaucracy",
        news_domains=(
            "reuters.com",
            "apnews.com",
            "bbc.com",
            "oireachtas.ie",
            "gov.uk",
            "energy.gov",
        ),
        bluesky_handles=(
            "reuters.com",
            "apnews.com",
        ),
        apply_status_quo_bias=True,
        default_shrink_alpha=0.78,
        market_weight=0.45,
        outside_view_hint=(
            "Government agencies and parliaments exhibit strong status-quo "
            "inertia: official root-cause investigations routinely take months, "
            "and opposition/private-member bills rarely pass Second Stage "
            "without a timed amendment or defeat."
        ),
    ),
}

_ELECTION_TERMS = (
    "election", "senate", "house race", "president", "governor", "ballot",
    "votes", "first-choice", "swing seat", "margin", "polling", "candidate",
    "primary", "runoff", "referendum",
)
_CONFLICT_TERMS = (
    "drone", "missile", "air force", "russia", "ukraine", "ceasefire",
    "invasion", "troops", "military", "airstrike", "nato", "gaza", "israel",
    "iran", "taiwan", "nuclear weapon", "sanctions",
)
_LAW_TERMS = (
    "court", "criminal", "indict", "federal case", "trial", "verdict",
    "supreme court", "lawsuit", "prosecut", "judge", "convict", "pending",
    "sentenc", "doj", "fbi",
)


def detect_politics_subtype(
    question_text: str,
    official_categories: list[str] | tuple[str, ...] = (),
) -> str:
    """Clasifica una pregunta de politics_geo_law en su subtipo operativo."""
    text = question_text.lower()
    cats = {c.lower() for c in official_categories}

    if "elections" in cats or any(term in text for term in _ELECTION_TERMS):
        return "elections"
    if any(term in text for term in _CONFLICT_TERMS):
        return "conflict_geo"
    if any(term in text for term in _LAW_TERMS):
        return "law_courts"
    if "geopolitics" in cats:
        return "conflict_geo"
    if "law" in cats and ("bill" not in text and "parliament" not in text and "dáil" not in text):
        return "law_courts"
    return "parliament_bureaucracy"
