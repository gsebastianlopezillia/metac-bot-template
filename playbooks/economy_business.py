"""Playbook operativo para la categoría economy_business (~27% del torneo).

Integra datos cuantitativos a 0 tokens (Yahoo Finance, FRED CSV oficial,
Polymarket, Manifold), prensa financiera curada y Alpaca/Benzinga News opcional.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import requests

from oracle_enrichment import fetch_polymarket_prior
from playbooks.finance_sources import (
    EquityStats,
    FredSeriesSummary,
    extract_fred_series_ids,
    extract_tickers,
    fetch_alpaca_news,
    fetch_fred_series,
    fetch_yahoo_stats,
)
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


@dataclass(frozen=True)
class EconomySubtypeSpec:
    """Configuración de fuentes y sesgos por subtipo económico/financiero."""

    subtype: str
    news_domains: tuple[str, ...]
    apply_status_quo_bias: bool
    default_shrink_alpha: float
    market_weight: float
    outside_view_hint: str


ECONOMY_SUBTYPES: dict[str, EconomySubtypeSpec] = {
    "equities_markets": EconomySubtypeSpec(
        subtype="equities_markets",
        news_domains=("reuters.com", "cnbc.com", "wsj.com", "marketwatch.com", "bloomberg.com"),
        apply_status_quo_bias=False,
        default_shrink_alpha=0.82,
        market_weight=0.55,
        outside_view_hint=(
            "Anchor on efficient-market random walk: over 1-14 day horizons, "
            "expected excess return vs. sector ETF is ~0% unless an earnings "
            "catalyst occurs. Scale numeric intervals directly from historical "
            "daily volatility (sigma * sqrt(trading_days))."
        ),
    ),
    "macro_fred_commodities": EconomySubtypeSpec(
        subtype="macro_fred_commodities",
        news_domains=("reuters.com", "bloomberg.com", "iea.org", "fred.stlouisfed.org", "cnbc.com"),
        apply_status_quo_bias=False,
        default_shrink_alpha=0.84,
        market_weight=0.50,
        outside_view_hint=(
            "Anchor directly on the latest official FRED / IEA / government "
            "release and its recent trend. Monthly macro and oil-demand "
            "reports exhibit strong revision inertia relative to the prior month."
        ),
    ),
    "earnings_filings": EconomySubtypeSpec(
        subtype="earnings_filings",
        news_domains=("reuters.com", "cnbc.com", "wsj.com", "sec.gov", "marketwatch.com"),
        apply_status_quo_bias=True,
        default_shrink_alpha=0.80,
        market_weight=0.50,
        outside_view_hint=(
            "For SEC 13F filings and quarterly guidance, compare against prior "
            "quarter holdings/guidance and Wall Street consensus. Large 13F "
            "positions change gradually unless public exits were already signaled."
        ),
    ),
    "corporate_events": EconomySubtypeSpec(
        subtype="corporate_events",
        news_domains=("reuters.com", "apnews.com", "cnbc.com", "bloomberg.com"),
        apply_status_quo_bias=True,
        default_shrink_alpha=0.78,
        market_weight=0.45,
        outside_view_hint=(
            "Corporate timelines, tenders, and hiring/selection schedules suffer "
            "from administrative delays. Apply status-quo inertia on short horizons."
        ),
    ),
}

_EQUITY_TERMS = (
    "stock", "etf", "outperform", "underperform", "market close", "trading day",
    "share price", "s&p 500", "nasdaq", "dow jones",
)
_MACRO_TERMS = (
    "fred", "gdp", "inflation", "cpi", "unemployment", "oil demand", "iea",
    "opec", "federal outlays", "interest rate", "fomc", "payroll", "recession",
)
_FILING_TERMS = (
    "13f", "sec filing", "guidance", "revenue", "earnings", "eps", "quarterly",
    "berkshire", "holdings",
)


def detect_economy_subtype(question_text: str, extra_text: str = "") -> str:
    """Clasifica una pregunta de economy_business en su subtipo cuantitativo."""
    combined = f"{question_text} {extra_text}".lower()
    if any(t in combined for t in _FILING_TERMS):
        return "earnings_filings"
    if any(t in combined for t in _EQUITY_TERMS):
        return "equities_markets"
    if any(t in combined for t in _MACRO_TERMS) or extract_fred_series_ids(f"{question_text} {extra_text}"):
        return "macro_fred_commodities"
    return "corporate_events"


def _format_economy_evidence(
    *,
    subtype: str,
    days_remaining: int | None,
    yahoo_stats: list[EquityStats],
    fred_summaries: list[FredSeriesSummary],
    polymarket_block: str,
    manifold_matches: list[MarketMatch],
    alpaca_news: list[NewsSnippet],
    news_items: list[NewsSnippet],
) -> str:
    sections: list[str] = [
        f"### Playbook Metadata\n- Domain: economy_business ({subtype})\n"
        f"- Estimated days to resolution: {days_remaining if days_remaining is not None else 'unknown'}"
    ]

    if yahoo_stats:
        y_lines = [
            f"- {s.ticker} (as of {s.last_date}): close=${s.last_close}, "
            f"14d={s.ret_14d_pct:+.2f}%, 30d={s.ret_30d_pct:+.2f}%, "
            f"daily_vol={s.daily_vol_pct:.2f}%, 10-trading-day 1σ≈±{s.expected_10d_move_pct:.2f}%"
            if s.ret_14d_pct is not None and s.ret_30d_pct is not None
            else f"- {s.ticker} (as of {s.last_date}): close=${s.last_close}, daily_vol={s.daily_vol_pct:.2f}%"
            for s in yahoo_stats
        ]
        sections.append("### [TIER 1 - LIVE MARKET DATA (YAHOO FINANCE)]\n" + "\n".join(y_lines))

    if fred_summaries:
        f_lines = []
        for fs in fred_summaries:
            pts = ", ".join(f"{d}={v:g}" for d, v in fs.observations)
            yoy = f" (last-step change: {fs.yoy_change_pct:+.2f}%)" if fs.yoy_change_pct is not None else ""
            f_lines.append(f"- FRED {fs.series_id}: {pts}{yoy}")
        sections.append("### [TIER 1 - OFFICIAL MACRO DATA (FRED)]\n" + "\n".join(f_lines))

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

    if alpaca_news:
        a_lines = [f"- [{n.published}] ({n.source}) {n.text}" for n in alpaca_news]
        sections.append("### [TIER 2 - ALPACA/BENZINGA FINANCIAL NEWS]\n" + "\n".join(a_lines))

    if news_items:
        n_lines = [f"- [{n.published}] ({n.source}) {n.text}" for n in news_items]
        sections.append(f"### [{news_items[0].tier}]\n" + "\n".join(n_lines))

    combined = "\n\n".join(sections).strip()
    if len(combined) > MAX_EVIDENCE_CHARS:
        return combined[: MAX_EVIDENCE_CHARS - 20].rstrip() + "\n[...truncated]"
    return combined


def build_economy_business_context(
    question_text: str,
    *,
    resolution_criteria: str = "",
    close_time: datetime | None = None,
    session: requests.Session | None = None,
    now: datetime | None = None,
) -> PlaybookContext:
    """Construye contexto cuantitativo completo para economy_business."""
    subtype = detect_economy_subtype(question_text, resolution_criteria)
    spec = ECONOMY_SUBTYPES[subtype]
    days_remaining = estimate_days_remaining(question_text, close_time=close_time, now=now)

    client = session or requests.Session()
    tickers = extract_tickers(question_text)
    fred_ids = extract_fred_series_ids(f"{question_text}\n{resolution_criteria}")

    yahoo_stats = [
        st for t in tickers if (st := fetch_yahoo_stats(t, session=client)) is not None
    ]
    fred_summaries = [
        fs for fid in fred_ids if (fs := fetch_fred_series(fid, session=client)) is not None
    ]
    polymarket_block = fetch_polymarket_prior(question_text, session=client)
    manifold_matches = fetch_manifold_prior(question_text, session=client)
    alpaca_news = fetch_alpaca_news(tickers, session=client)
    news_items = fetch_google_news_rss(
        question_text,
        domains=spec.news_domains,
        session=client,
    )

    market_prior, market_source = _extract_primary_market_prior(
        polymarket_block, manifold_matches
    )
    evidence_text = _format_economy_evidence(
        subtype=subtype,
        days_remaining=days_remaining,
        yahoo_stats=yahoo_stats,
        fred_summaries=fred_summaries,
        polymarket_block=polymarket_block,
        manifold_matches=manifold_matches,
        alpaca_news=alpaca_news,
        news_items=news_items,
    )
    prompt_addendum = (
        "### Domain Forecasting Protocol (economy_business)\n"
        f"- Subtype guidance ({spec.subtype}): {spec.outside_view_hint}\n"
        "- Quantitative discipline: anchor on TIER 1 numbers (Yahoo volatility, "
        "FRED series, consensus/markets) rather than narrative speculation.\n"
        "- Pre-mortem required: before finalizing numbers, state the single most "
        "plausible market shock, revision, or guidance surprise that could invalidate your forecast."
    )

    return PlaybookContext(
        playbook="economy_business",
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
