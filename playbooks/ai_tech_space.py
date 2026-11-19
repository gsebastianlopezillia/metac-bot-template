"""Playbook operativo para la categoría ai_tech_space (~6% del torneo).

Integra Launch Library 2 (SpaceX, Blue Origin, Virgin, JAXA/Japón, ISRO/India,
CNSA/China), GitHub Public API, ClinicalTrials.gov (si aplica), Polymarket,
Manifold y prensa espacial/tecnológica curada.
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
from playbooks.science_sources import ClinicalTrialsSummary, fetch_clinical_trials_summary
from playbooks.tech_space_sources import (
    GitHubRepoSummary,
    LaunchRecord,
    detect_space_provider,
    extract_github_repos,
    fetch_github_repo_summary,
    fetch_launch_library_summary,
)


@dataclass(frozen=True)
class TechSpaceSpec:
    """Configuración de fuentes y sesgos por subtipo de IA, tech y espacio."""

    subtype: str
    news_domains: tuple[str, ...]
    apply_status_quo_bias: bool
    default_shrink_alpha: float
    market_weight: float
    outside_view_hint: str


TECH_SPACE_SUBTYPES: dict[str, TechSpaceSpec] = {
    "space_launches": TechSpaceSpec(
        "space_launches",
        ("spacenews.com", "nasaspaceflight.com", "spaceflightnow.com", "global.jaxa.jp", "isro.gov.in", "reuters.com", "arstechnica.com"),
        True, 0.82, 0.50,
        "For mature cadence vehicles (Falcon 9, Long March), anchor on empirical 30-day launch cadence from Launch Library 2. For developmental rockets (Starship, New Glenn, H3, Gaganyaan, Virgin), apply strong delay/scrub bias.",
    ),
    "ai_models_releases": TechSpaceSpec(
        "ai_models_releases",
        ("anthropic.com", "openai.com", "deepmind.google", "techcrunch.com", "theverge.com", "arstechnica.com", "reuters.com"),
        True, 0.80, 0.55,
        "AI lab model releases on short deadlines suffer frequent safety/compute delays unless an official launch event or API preview is already live. Apply status-quo bias toward NO on tight windows.",
    ),
    "github_open_source": TechSpaceSpec(
        "github_open_source",
        ("github.com", "arstechnica.com", "techcrunch.com", "reuters.com"),
        False, 0.82, 0.45,
        "Check recent GitHub commit velocity and last push date to gauge active throughput before extrapolating repository counts or milestones.",
    ),
    "tech_dashboards_general": TechSpaceSpec(
        "tech_dashboards_general",
        ("arstechnica.com", "techcrunch.com", "wired.com", "theverge.com", "reuters.com"),
        False, 0.82, 0.45,
        "Anchor on the latest verifiable dashboard count (e.g. ClinicalTrials.gov, DeFlock) and extrapolate smooth recent growth with wide uncertainty bands.",
    ),
}

_SPACE_TERMS = (
    "spacex", "falcon", "starship", "blue origin", "new glenn", "virgin galactic",
    "jaxa", "isro", "cnsa", "casc", "long march", "orbital launch", "rocket",
    "satellite", "moon", "mars", "nasa", "tiangong",
)
_AI_RELEASE_TERMS = (
    "anthropic", "claude", "openai", "gpt-", "gemini", "deepmind", "llama",
    "mistral", "xai", "grok", "publicly released", "frontier model", "benchmark",
)


def detect_tech_space_subtype(question_text: str) -> str:
    """Clasifica una pregunta de ai_tech_space en su subtipo operativo."""
    text = question_text.lower()
    if detect_space_provider(question_text) or any(t in text for t in _SPACE_TERMS):
        return "space_launches"
    if extract_github_repos(question_text) or "repository" in text or "github" in text:
        return "github_open_source"
    if any(t in text for t in _AI_RELEASE_TERMS):
        return "ai_models_releases"
    return "tech_dashboards_general"


def _format_tech_space_evidence(
    *,
    subtype: str,
    days_remaining: int | None,
    prev_launches: list[LaunchRecord],
    next_launches: list[LaunchRecord],
    github_summaries: list[GitHubRepoSummary],
    trials_summary: ClinicalTrialsSummary | None,
    polymarket_block: str,
    manifold_matches: list[MarketMatch],
    news_items: list[NewsSnippet],
) -> str:
    sections: list[str] = [
        f"### Playbook Metadata\n- Domain: ai_tech_space ({subtype})\n"
        f"- Estimated days to resolution: {days_remaining if days_remaining is not None else 'unknown'}"
    ]
    if prev_launches or next_launches:
        l_lines: list[str] = []
        for lr in prev_launches:
            l_lines.append(f"- [PAST {lr.date}] {lr.name} ({lr.provider}) status={lr.status}")
        for lr in next_launches:
            l_lines.append(f"- [UPCOMING {lr.date}] {lr.name} ({lr.provider}) status={lr.status}")
        sections.append("### [TIER 1 - LAUNCH LIBRARY 2 (GLOBAL SPACE CADENCE)]\n" + "\n".join(l_lines))
    if github_summaries:
        g_lines: list[str] = []
        for g in github_summaries:
            g_lines.append(f"- Repo {g.slug}: stars={g.stars}, last_push={g.pushed_at} | {g.description}")
            for cm in g.recent_commits:
                g_lines.append(f"  * commit {cm}")
        sections.append("### [TIER 1 - GITHUB PUBLIC API]\n" + "\n".join(g_lines))
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
    if news_items:
        n_lines = [f"- [{n.published}] ({n.source}) {n.text}" for n in news_items]
        sections.append(f"### [{news_items[0].tier}]\n" + "\n".join(n_lines))

    combined = "\n\n".join(sections).strip()
    if len(combined) > MAX_EVIDENCE_CHARS:
        return combined[: MAX_EVIDENCE_CHARS - 20].rstrip() + "\n[...truncated]"
    return combined


def build_ai_tech_space_context(
    question_text: str,
    *,
    close_time: datetime | None = None,
    session: requests.Session | None = None,
    now: datetime | None = None,
) -> PlaybookContext:
    """Construye contexto completo para ai_tech_space (SpaceX/BO/Virgin/JAXA/ISRO/China + IA)."""
    subtype = detect_tech_space_subtype(question_text)
    spec = TECH_SPACE_SUBTYPES[subtype]
    days_remaining = estimate_days_remaining(question_text, close_time=close_time, now=now)

    client = session or requests.Session()
    prev_launches, next_launches = (
        fetch_launch_library_summary(question_text, session=client)
        if subtype == "space_launches"
        else ([], [])
    )
    repo_slugs = extract_github_repos(question_text)
    github_summaries = [
        gs for slug in repo_slugs if (gs := fetch_github_repo_summary(slug, session=client)) is not None
    ]
    trials_summary = (
        fetch_clinical_trials_summary(
            question_text,
            session=client,
            recruiting_only="recruiting" in question_text.lower(),
        )
        if "clinicaltrials" in question_text.lower()
        else None
    )
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
    evidence_text = _format_tech_space_evidence(
        subtype=subtype,
        days_remaining=days_remaining,
        prev_launches=prev_launches,
        next_launches=next_launches,
        github_summaries=github_summaries,
        trials_summary=trials_summary,
        polymarket_block=polymarket_block,
        manifold_matches=manifold_matches,
        news_items=news_items,
    )
    prompt_addendum = (
        "### Domain Forecasting Protocol (ai_tech_space)\n"
        f"- Subtype guidance ({spec.subtype}): {spec.outside_view_hint}\n"
        "- Technical discipline: trust live Launch Library 2 / GitHub / ClinicalTrials "
        "telemetry over hype or rumor blogs.\n"
        "- Pre-mortem required: note weather scrubs, regulatory holds, or quiet "
        "release delays that could invalidate your forecast."
    )

    return PlaybookContext(
        playbook="ai_tech_space",
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
