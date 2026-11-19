"""Playbook operativo para la categoría climate_environment (~6% del torneo).

Integra observaciones físicas oficiales (NOAA CPC semanal + ONI trimestral,
US Drought Monitor CONUS), mercados de predicción y prensa científica/regional.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import requests

from oracle_enrichment import fetch_noaa_enso_context, fetch_polymarket_prior
from playbooks.climate_sources import (
    DroughtReading,
    OniReading,
    fetch_noaa_oni_latest,
    fetch_us_drought_latest,
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
from playbooks.science_sources import fetch_regional_news_rss


@dataclass(frozen=True)
class ClimateSubtypeSpec:
    """Configuración de fuentes y sesgos por subtipo climático/ambiental."""

    subtype: str
    news_domains: tuple[str, ...]
    apply_status_quo_bias: bool
    default_shrink_alpha: float
    market_weight: float
    outside_view_hint: str


CLIMATE_SUBTYPES: dict[str, ClimateSubtypeSpec] = {
    "enso_weather_extremes": ClimateSubtypeSpec(
        "enso_weather_extremes",
        ("noaa.gov", "nhc.noaa.gov", "copernicus.eu", "reuters.com", "apnews.com"),
        False, 0.84, 0.50,
        "Anchor directly on official NOAA CPC ONI/weekly Niño 3.4 readings and seasonal climatology. Ocean indices and late-season hurricane ACE evolve with strong physical persistence.",
    ),
    "drought_hydrology": ClimateSubtypeSpec(
        "drought_hydrology",
        ("droughtmonitor.unl.edu", "noaa.gov", "reuters.com", "apnews.com"),
        False, 0.85, 0.45,
        "US Drought Monitor percentages (D1+ = in drought, D0+ = abnormally dry or worse) change slowly week-to-week (typically +-1 to 3 percentage points). Anchor tightly on the latest CONUS observation.",
    ),
    "energy_emissions": ClimateSubtypeSpec(
        "energy_emissions",
        ("reuters.com", "gie.eu", "iea.org", "bloomberg.com", "euractiv.com"),
        False, 0.84, 0.50,
        "EU gas storage and energy balance metrics move along smooth seasonal injection/withdrawal curves. Anchor on latest reported filling levels and daily withdrawal rates.",
    ),
    "climate_diplomacy_ecology": ClimateSubtypeSpec(
        "climate_diplomacy_ecology",
        ("reuters.com", "apnews.com", "bbc.com", "carbonbrief.org", "unep.org"),
        True, 0.80, 0.45,
        "International climate summits (COP) almost always run into overtime rather than suspending without completing agendas. For wildlife/pest dashboards (e.g. Senasica), anchor on recent regional reports.",
    ),
}

_ENSO_WEATHER_TERMS = (
    "el niño", "el nino", "la niña", "la nina", "enso", "hurricane", "cyclone",
    "typhoon", "ace", "accumulated cyclone", "temperature", "warming", "noaa", "heatwave",
)
_DROUGHT_TERMS = ("drought", "droughtmonitor", "dryness", "reservoir", "rainfall", "precipitation")
_ENERGY_TERMS = ("gas storage", "agsi", "lng", "emissions", "co2", "renewable", "solar", "wind power", "carbon")


def detect_climate_subtype(question_text: str) -> str:
    """Clasifica una pregunta de climate_environment en su subtipo físico/ambiental."""
    text = question_text.lower()
    if any(t in text for t in _DROUGHT_TERMS):
        return "drought_hydrology"
    if any(t in text for t in _ENSO_WEATHER_TERMS):
        return "enso_weather_extremes"
    if any(t in text for t in _ENERGY_TERMS):
        return "energy_emissions"
    return "climate_diplomacy_ecology"


def _format_climate_evidence(
    *,
    subtype: str,
    days_remaining: int | None,
    noaa_weekly: str,
    oni_readings: list[OniReading],
    drought_readings: list[DroughtReading],
    polymarket_block: str,
    manifold_matches: list[MarketMatch],
    regional_news: list[NewsSnippet],
    news_items: list[NewsSnippet],
) -> str:
    sections: list[str] = [
        f"### Playbook Metadata\n- Domain: climate_environment ({subtype})\n"
        f"- Estimated days to resolution: {days_remaining if days_remaining is not None else 'unknown'}"
    ]
    if noaa_weekly:
        sections.append(noaa_weekly)
    if oni_readings:
        oni_str = ", ".join(f"{r.season} {r.year}={r.anomaly:+.2f}°C" for r in oni_readings)
        sections.append(f"### [TIER 1 - OFFICIAL NOAA CPC ONI (3-MONTH INDEX)]\n- Recent ONI anomalies: {oni_str}")
    if drought_readings:
        d_lines = [
            f"- Map {r.map_date}: In Drought (D1+)={r.d1_drought_pct:.2f}%, "
            f"Abnormally Dry+ (D0+)={r.d0_or_worse_pct:.2f}%, Severe+ (D2+)={r.d2_severe_pct:.2f}%, None={r.none_pct:.2f}%"
            for r in drought_readings
        ]
        sections.append("### [TIER 1 - OFFICIAL US DROUGHT MONITOR (CONUS)]\n" + "\n".join(d_lines))
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


def build_climate_environment_context(
    question_text: str,
    *,
    close_time: datetime | None = None,
    session: requests.Session | None = None,
    now: datetime | None = None,
) -> PlaybookContext:
    """Construye contexto completo (observaciones físicas + prensa) para climate_environment."""
    subtype = detect_climate_subtype(question_text)
    spec = CLIMATE_SUBTYPES[subtype]
    days_remaining = estimate_days_remaining(question_text, close_time=close_time, now=now)

    client = session or requests.Session()
    lower = question_text.lower()
    is_enso = any(k in lower for k in ("niño", "nino", "niña", "nina", "enso", "oni"))

    noaa_weekly = fetch_noaa_enso_context(session=client) if is_enso else ""
    oni_readings = fetch_noaa_oni_latest(session=client) if is_enso else []
    drought_readings = (
        fetch_us_drought_latest(session=client, now=now)
        if subtype == "drought_hydrology"
        else []
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
    evidence_text = _format_climate_evidence(
        subtype=subtype,
        days_remaining=days_remaining,
        noaa_weekly=noaa_weekly,
        oni_readings=oni_readings,
        drought_readings=drought_readings,
        polymarket_block=polymarket_block,
        manifold_matches=manifold_matches,
        regional_news=regional_news,
        news_items=news_items,
    )
    prompt_addendum = (
        "### Domain Forecasting Protocol (climate_environment)\n"
        f"- Subtype guidance ({spec.subtype}): {spec.outside_view_hint}\n"
        "- Physical discipline: verify exact index definitions (e.g. D1+ vs D0+ in "
        "US Drought Monitor, 3-month ONI vs weekly Niño 3.4) before finalizing numbers.\n"
        "- Pre-mortem required: note weather-regime shifts, data revisions, or "
        "diplomatic deadlocks that could invalidate your forecast."
    )

    return PlaybookContext(
        playbook="climate_environment",
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
