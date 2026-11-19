"""Playbook operativo para la categoría crypto (~4% del torneo).

Reutiliza Yahoo Finance (`BTC-USD`, `ETH-USD`, `ZEC-USD`, etc.) para precio y
volatilidad histórica diaria, calcula probabilidades log-normales de umbral o
toque de barrera a 0 tokens, y combina con Polymarket/Manifold y prensa crypto.
"""

from __future__ import annotations

from datetime import datetime
import math
import re

import requests

from oracle_enrichment import fetch_polymarket_prior
from playbooks.finance_sources import EquityStats, fetch_yahoo_stats
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

CRYPTO_NEWS_DOMAINS = (
    "coindesk.com",
    "theblock.co",
    "cointelegraph.com",
    "reuters.com",
    "bloomberg.com",
)

_CRYPTO_MAP: dict[str, str] = {
    "bitcoin": "BTC-USD",
    "btc": "BTC-USD",
    "ethereum": "ETH-USD",
    "eth": "ETH-USD",
    "solana": "SOL-USD",
    "sol": "SOL-USD",
    "zcash": "ZEC-USD",
    "zec": "ZEC-USD",
    "ripple": "XRP-USD",
    "xrp": "XRP-USD",
    "dogecoin": "DOGE-USD",
    "doge": "DOGE-USD",
    "cardano": "ADA-USD",
    "chainlink": "LINK-USD",
    "bnb": "BNB-USD",
}

_USD_TARGET_RE = re.compile(r"\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)")


def extract_crypto_tickers(question_text: str, max_tickers: int = 2) -> list[str]:
    """Detecta pares USD de criptomonedas soportados por Yahoo Finance."""
    tokens = set(re.findall(r"[a-zA-Z0-9]+", question_text.lower()))
    found: list[str] = []
    for key, yahoo_sym in _CRYPTO_MAP.items():
        if key in tokens and yahoo_sym not in found:
            found.append(yahoo_sym)
    return found[:max_tickers]


def extract_usd_target(question_text: str) -> float | None:
    """Extrae un umbral de precio en USD (ej. $85,000 -> 85000.0)."""
    match = _USD_TARGET_RE.search(question_text)
    if not match:
        return None
    try:
        val = float(match.group(1).replace(",", ""))
        return val if val > 0 else None
    except ValueError:
        return None


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def estimate_lognormal_threshold_prob(
    spot: float,
    target: float,
    daily_vol_pct: float,
    days_remaining: int,
    *,
    is_touch_barrier: bool = False,
) -> float | None:
    """Estima P(S_T >= K) o probabilidad de toque bajo random walk log-normal (0 drift)."""
    if spot <= 0 or target <= 0 or daily_vol_pct <= 0:
        return None
    horizon = max(1, days_remaining)
    sigma_t = (daily_vol_pct / 100.0) * math.sqrt(float(horizon))
    if sigma_t <= 1e-6:
        return 1.0 if spot >= target else 0.0

    if is_touch_barrier:
        if spot >= target:
            return 0.99
        z = -math.log(target / spot) / sigma_t
        return min(0.99, max(0.01, 2.0 * _norm_cdf(z)))

    z = math.log(spot / target) / sigma_t
    return min(0.99, max(0.01, _norm_cdf(z)))


def _detect_crypto_subtype(question_text: str) -> str:
    lower = question_text.lower()
    if any(w in lower for w in ("reach", "touch", "hit", "at or above", "at or below", "price", "closing price")):
        return "price_volatility"
    return "protocol_regulatory"


def _format_crypto_evidence(
    *,
    subtype: str,
    days_remaining: int | None,
    stats_list: list[EquityStats],
    quant_note: str,
    polymarket_block: str,
    manifold_matches: list[MarketMatch],
    news_items: list[NewsSnippet],
) -> str:
    sections: list[str] = [
        f"### Playbook Metadata\n- Domain: crypto ({subtype})\n"
        f"- Estimated days to resolution: {days_remaining if days_remaining is not None else 'unknown'}"
    ]
    if stats_list:
        c_lines = [
            f"- {s.ticker} (as of {s.last_date}): spot=${s.last_close:,.2f}, "
            f"14d={s.ret_14d_pct:+.2f}%, 30d={s.ret_30d_pct:+.2f}%, "
            f"daily_vol={s.daily_vol_pct:.2f}%"
            if s.ret_14d_pct is not None and s.ret_30d_pct is not None
            else f"- {s.ticker} (as of {s.last_date}): spot=${s.last_close:,.2f}, daily_vol={s.daily_vol_pct:.2f}%"
            for s in stats_list
        ]
        if quant_note:
            c_lines.append(f"- {quant_note}")
        sections.append("### [TIER 1 - LIVE CRYPTO SPOT & VOLATILITY (YAHOO FINANCE)]\n" + "\n".join(c_lines))
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


def build_crypto_context(
    question_text: str,
    *,
    close_time: datetime | None = None,
    session: requests.Session | None = None,
    now: datetime | None = None,
) -> PlaybookContext:
    """Construye contexto cuantitativo completo para preguntas de criptomonedas."""
    subtype = _detect_crypto_subtype(question_text)
    days_remaining = estimate_days_remaining(question_text, close_time=close_time, now=now)

    client = session or requests.Session()
    tickers = extract_crypto_tickers(question_text)
    stats_list = [
        st for t in tickers if (st := fetch_yahoo_stats(t, session=client)) is not None
    ]

    quant_note = ""
    quant_prior: float | None = None
    target_usd = extract_usd_target(question_text)
    if stats_list and target_usd is not None and stats_list[0].daily_vol_pct:
        s0 = stats_list[0]
        is_touch = any(w in question_text.lower() for w in ("reach", "touch", "hit"))
        est = estimate_lognormal_threshold_prob(
            s0.last_close,
            target_usd,
            s0.daily_vol_pct,
            days_remaining if days_remaining is not None else 7,
            is_touch_barrier=is_touch,
        )
        if est is not None:
            mode_lbl = "barrier-touch" if is_touch else "close-above"
            quant_note = (
                f"Log-normal zero-drift {mode_lbl} baseline for {s0.ticker} >= ${target_usd:,.2f}: "
                f"{est:.1%} (using daily_vol={s0.daily_vol_pct:.2f}%)."
            )
            quant_prior = est

    polymarket_block = fetch_polymarket_prior(question_text, session=client)
    manifold_matches = fetch_manifold_prior(question_text, session=client)
    news_items = fetch_google_news_rss(
        question_text,
        domains=CRYPTO_NEWS_DOMAINS,
        session=client,
    )

    market_prior, market_source = _extract_primary_market_prior(
        polymarket_block, manifold_matches
    )
    if market_prior is None and quant_prior is not None and days_remaining is not None and days_remaining > 0:
        market_prior = quant_prior
        market_source = "LogNormalVolBaseline"

    evidence_text = _format_crypto_evidence(
        subtype=subtype,
        days_remaining=days_remaining,
        stats_list=stats_list,
        quant_note=quant_note,
        polymarket_block=polymarket_block,
        manifold_matches=manifold_matches,
        news_items=news_items,
    )
    prompt_addendum = (
        "### Domain Forecasting Protocol (crypto)\n"
        "- Quantitative anchor: crypto returns are fat-tailed and close to a "
        "random walk over short horizons. Anchor strictly on TIER 1 spot price, "
        "historical daily volatility, log-normal baseline, and Polymarket odds.\n"
        "- Numeric discipline: use wide fat-tailed intervals (crypto routinely "
        "moves 2-3x normal equity volatility) and avoid narrative hype.\n"
        "- Pre-mortem required: note macro liquidity shocks, ETF flows, or "
        "exchange/regulatory catalysts that could trigger tail moves."
    )

    return PlaybookContext(
        playbook="crypto",
        subtype=subtype,
        days_remaining=days_remaining,
        market_prior=market_prior,
        market_prior_source=market_source,
        apply_status_quo_bias=False,
        default_shrink_alpha=0.82,
        market_weight=0.55,
        evidence_text=evidence_text,
        prompt_addendum=prompt_addendum,
    )
