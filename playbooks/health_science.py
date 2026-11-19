"""Playbook operativo para la categoría health_science (~11% del torneo).

Cubre Health & Pandemics + Natural Sciences usando APIs abiertas (arXiv,
ClinicalTrials.gov, Polymarket, Manifold) y prensa científica/regional curada.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import requests

from oracle_enrichment import fetch_polymarket_prior
from playbooks.open_sources import (
    MarketMatch,
    NewsSnippet,
    fetch_google_news_rss,
    fetch_manifold_prior,
)
from playbooks.politics_geo_law import (
    MAX_EVIDENCE_CHARS,
    PlaybookContext,
    _extract_primary_market_prior,
    estimate_days_remaining,
)
from playbooks.science_sources import (
    ArxivPaper,
    ClinicalTrialsSummary,
    fetch_arxiv_preprints,
    fetch_clinical_trials_summary,
    fetch_regional_news_rss,
)


@dataclass(frozen=True)
class HealthScienceSpec:
    """Configuración de fuentes y sesgos por subtipo científico/sanitario."""

    subtype: str
    news_domains: tuple[str, ...]
    apply_status_quo_bias: bool
    default_shrink_alpha: float
    market_weight: float
    outside_view_hint: str


HEALTH_SCIENCE_SUBTYPES: dict[str, HealthScienceSpec] = {
    "epidemiology_outbreaks": HealthScienceSpec(
        subtype="epidemiology_outbreaks",
        news_domains=("who.int", "cdc.gov", "cidrap.umn.edu", "reliefweb.int", "reuters.com", "apnews.com"),
        apply_status_quo_bias=False,
        default_shrink_alpha=0.82,
        market_weight=0.45,
        outside_view_hint=(
            "Epidemiological bulletins exhibit strong week-over-week autocorrelation "
            "and reporting-lag noise. Anchor on the most recent official surveillance "
            "trend and use fat-tailed numeric intervals for daily/weekly case counts."
        ),
    ),
    "clinical_regulatory": HealthScienceSpec(
        subtype="clinical_regulatory",
        news_domains=("fda.gov", "statnews.com", "fiercepharma.com", "reuters.com", "nature.com"),
        apply_status_quo_bias=True,
        default_shrink_alpha=0.80,
        market_weight=0.50,
        outside_view_hint=(
            "Regulatory approvals and clinical trial milestones face frequent "
            "procedural delays. Check ClinicalTrials.gov status and base rates "
            "before assigning high probability to short-horizon breakthroughs."
        ),
    ),
    "science_preprints": HealthScienceSpec(
        subtype="science_preprints",
        news_domains=("nature.com", "science.org", "quantamagazine.org", "phys.org", "arxiv.org"),
        apply_status_quo_bias=True,
        default_shrink_alpha=0.78,
        market_weight=0.45,
        outside_view_hint=(
            "Major theoretical breakthroughs, proofs, or manuscripts on tight "
            "deadlines have a very low base rate unless a preprint is already "
            "visible on arXiv or formally announced with a release date."
        ),
    ),
    "local_health_ops": HealthScienceSpec(
        subtype="local_health_ops",
        news_domains=("reuters.com", "apnews.com", "bbc.com"),
        apply_status_quo_bias=True,
        default_shrink_alpha=0.80,
        market_weight=0.45,
        outside_view_hint=(
            "For local hospital operations or wildlife tracking, weigh regional "
            "press directly and apply continuity/status-quo bias unless an active "
            "closure, reopening date, or recent telemetry ping is confirmed."
        ),
    ),
}

_EPIDEMIO_TERMS = (
    "measles", "ebola", "outbreak", "cases", "infogripe", "srag", "epidemiological",
    "pandemic", "flu", "influenza", "dengue", "mpox", "cholera", "h5n1", "who ",
    "cdc ", "dghs", "bulletin",
)
_CLINICAL_TERMS = (
    "fda", "clinical trial", "clinicaltrials", "phase 1", "phase 2", "phase 3",
    "vaccine", "drug approval", "ema ", "recruiting", "therapy",
)
_PREPRINT_TERMS = (
    "manuscript", "arxiv", "preprint", "proving", "theorem", "yang", "mills",
    "conjecture", "physics", "paper", "journal", "nature", "fusion", "superconductor",
)


def detect_health_science_subtype(question_text: str) -> str:
    """Clasifica una pregunta de health_science en su subtipo operativo."""
    text = question_text.lower()
    if any(t in text for t in _EPIDEMIO_TERMS):
        return "epidemiology_outbreaks"
    if any(t in text for t in _CLINICAL_TERMS):
        return "clinical_regulatory"
    if any(t in text for t in _PREPRINT_TERMS):
        return "science_preprints"
    return "local_health_ops"


def _format_health_evidence(
    *,
    subtype: str,
    days_remaining: int | None,
    arxiv_papers: list[ArxivPaper],
    trials_summary: ClinicalTrialsSummary | None,
    polymarket_block: str,
    manifold_matches: list[MarketMatch],
    regional_news: list[NewsSnippet],
    news_items: list[NewsSnippet],
) -> str:
    sections: list[str] = [
        f"### Playbook Metadata\n- Domain: health_science ({subtype})\n"
        f"- Estimated days to resolution: {days_remaining if days_remaining is not None else 'unknown'}"
    ]

    if arxiv_papers:
        a_lines = [
            f"- [{p.published}] {p.title} ({', '.join(p.authors)})"
            for p in arxiv_papers
        ]
        sections.append("### [TIER 1 - ARXIV PREPRINTS]\n" + "\n".join(a_lines))

    if trials_summary is not None:
        t_lines = [
            f"- Query '{trials_summary.query}': totalCount={trials_summary.total_count}",
            *[f"  * {title}" for title in trials_summary.sample_titles],
        ]
        sections.append("### [TIER 1 - CLINICALTRIALS.GOV V2]\n" + "\n".join(t_lines))

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

    if regional_news:
        r_lines = [f"- [{n.published}] ({n.source}) {n.text}" for n in regional_news]
        sections.append(f"### [{regional_news[0].tier}]\n" + "\n".join(r_lines))

    if news_items:
        n_lines = [f"- [{n.published}] ({n.source}) {n.text}" for n in news_items]
        sections.append(f"### [{news_items[0].tier}]\n" + "\n".join(n_lines))

    combined = "\n\n".join(sections).strip()
    if len(combined) > MAX_EVIDENCE_CHARS:
        return combined[: MAX_EVIDENCE_CHARS - 20].rstrip() + "\n[...truncated]"
    return combined


def build_health_science_context(
    question_text: str,
    *,
    close_time: datetime | None = None,
    session: requests.Session | None = None,
    now: datetime | None = None,
) -> PlaybookContext:
    """Construye contexto completo (fuentes científicas/regionales + sesgos)."""
    subtype = detect_health_science_subtype(question_text)
    spec = HEALTH_SCIENCE_SUBTYPES[subtype]
    days_remaining = estimate_days_remaining(question_text, close_time=close_time, now=now)

    client = session or requests.Session()
    arxiv_papers = (
        fetch_arxiv_preprints(question_text, session=client)
        if subtype == "science_preprints"
        else []
    )
    trials_summary = (
        fetch_clinical_trials_summary(
            question_text,
            session=client,
            recruiting_only="recruiting" in question_text.lower(),
        )
        if subtype == "clinical_regulatory"
        else None
    )
    polymarket_block = fetch_polymarket_prior(question_text, session=client)
    manifold_matches = fetch_manifold_prior(question_text, session=client)
    regional_news = fetch_regional_news_rss(question_text, session=client)
    news_items = fetch_google_news_rss(
        question_text,
        domains=spec.news_domains,
        session=client,
    )

    market_prior, market_source = _extract_primary_market_prior(
        polymarket_block, manifold_matches
    )
    evidence_text = _format_health_evidence(
        subtype=subtype,
        days_remaining=days_remaining,
        arxiv_papers=arxiv_papers,
        trials_summary=trials_summary,
        polymarket_block=polymarket_block,
        manifold_matches=manifold_matches,
        regional_news=regional_news,
        news_items=news_items,
    )
    prompt_addendum = (
        "### Domain Forecasting Protocol (health_science)\n"
        f"- Subtype guidance ({spec.subtype}): {spec.outside_view_hint}\n"
        "- Evidence rule: prioritize official health bulletins, arXiv/ClinicalTrials "
        "records, and local-language regional press over generic speculation.\n"
        "- Pre-mortem required: state the most plausible reporting delay, revision, "
        "or operational reversal that could invalidate your forecast."
    )

    return PlaybookContext(
        playbook="health_science",
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
