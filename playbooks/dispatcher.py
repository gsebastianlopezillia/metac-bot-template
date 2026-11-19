"""Despachador universal de playbooks y calibración para FallTemplateBot2026.

Desacopla main.py de los playbooks individuales: cada nueva categoría afinada
se registra aquí sin volver a modificar el bot principal.
"""

from __future__ import annotations

from typing import Any

from calibration import (
    BinaryCalibrationResult,
    calibrate_binary_forecast,
    calibrate_multichoice_probs,
    widen_numeric_percentiles,
)

CORE_FORECASTING_RULES = """### Core Forecasting Discipline (Mandatory)
1. **Tier-1 Evidence Supremacy:** Never contradict live TIER 1 data (Yahoo Finance, FRED, NOAA, US Drought Monitor, Launch Library 2, The Odds API, GitHub, ClinicalTrials, Polymarket/Manifold) with outdated training memory.
2. **Historical Memory Check ("Viejo Memorioso" / Reference Class):** Before finalizing your numbers, recall 2-3 real historical precedents of similar human or institutional situations (political standoffs, bureaucratic delays, market overreactions, diplomatic summits, legal timelines). Explicitly weigh how often pure textbook logic lost to human inertia, ego, panic, or muddling-through — and pull your estimate toward the historical base rate.
3. **Precision & Calibration:** Verify exact units, definitions, and resolution deadlines. Avoid lazy 50% defaults, avoid >95% or <5% extremes unless already decided, and keep numeric P10-P90 intervals wide enough for real-world fat tails."""
from category_router import classify_metaculus_question
from oracle_enrichment import build_external_enrichment
from playbooks.ai_tech_space import build_ai_tech_space_context
from playbooks.climate_environment import build_climate_environment_context
from playbooks.crypto import build_crypto_context
from playbooks.economy_business import build_economy_business_context
from playbooks.health_science import build_health_science_context
from playbooks.politics_geo_law import PlaybookContext, build_politics_geo_law_context
from playbooks.sports_entertainment import build_sports_entertainment_context


def route_and_build_context(question: Any) -> tuple[PlaybookContext | None, str]:
    """Clasifica la pregunta y construye su contexto externo (0 tokens LLM).

    Devuelve (PlaybookContext | None, texto_evidencia). Si la categoría aún no
    tiene playbook dedicado, cae al enriquecimiento genérico sin romper nada.
    """
    question_text = str(getattr(question, "question_text", "") or "")
    resolution_criteria = str(getattr(question, "resolution_criteria", "") or "")
    close_time = getattr(question, "close_time", None)

    decision = classify_metaculus_question(question)
    if decision.playbook == "politics_geo_law":
        ctx = build_politics_geo_law_context(
            question_text,
            official_categories=decision.official_categories,
            close_time=close_time,
        )
        return ctx, ctx.evidence_text

    if decision.playbook == "economy_business":
        ctx = build_economy_business_context(
            question_text,
            resolution_criteria=resolution_criteria,
            close_time=close_time,
        )
        return ctx, ctx.evidence_text

    if decision.playbook == "health_science":
        ctx = build_health_science_context(
            question_text,
            close_time=close_time,
        )
        return ctx, ctx.evidence_text

    if decision.playbook == "sports_entertainment":
        ctx = build_sports_entertainment_context(
            question_text,
            close_time=close_time,
        )
        return ctx, ctx.evidence_text

    if decision.playbook == "climate_environment":
        ctx = build_climate_environment_context(
            question_text,
            close_time=close_time,
        )
        return ctx, ctx.evidence_text

    if decision.playbook == "ai_tech_space":
        ctx = build_ai_tech_space_context(
            question_text,
            close_time=close_time,
        )
        return ctx, ctx.evidence_text

    if decision.playbook == "crypto":
        ctx = build_crypto_context(
            question_text,
            close_time=close_time,
        )
        return ctx, ctx.evidence_text

    fallback_text = build_external_enrichment(question_text)
    return None, fallback_text


def calibrate_binary_with_context(
    raw_probability: float,
    ctx: PlaybookContext | None,
) -> BinaryCalibrationResult:
    """Aplica calibración binaria usando los parámetros del playbook o defaults seguros."""
    if ctx is None:
        return calibrate_binary_forecast(
            raw_probability,
            shrink_alpha=0.85,
            low_clamp=0.02,
            high_clamp=0.98,
        )
    return calibrate_binary_forecast(
        raw_probability,
        market_prior=ctx.market_prior,
        market_weight=ctx.market_weight,
        shrink_alpha=ctx.default_shrink_alpha,
        days_remaining=ctx.days_remaining,
        apply_status_quo=ctx.apply_status_quo_bias,
        low_clamp=0.03,
        high_clamp=0.97,
    )


def _to_numeric_bound(raw_bound: Any) -> float | None:
    if raw_bound is None:
        return None
    if isinstance(raw_bound, (int, float)):
        return float(raw_bound)
    if hasattr(raw_bound, "timestamp"):
        return float(raw_bound.timestamp())
    return None


def widen_percentile_objects(
    percentile_list: list[Any],
    question: Any,
    *,
    inflation_factor: float = 1.25,
) -> list[Any]:
    """Ensancha una lista de objetos Percentile de forecasting_tools preservando límites."""
    if len(percentile_list) < 2:
        return percentile_list

    # forecasting_tools usa Percentile.percentile en (0, 1) (ej. 0.10, 0.20, ..., 0.90)
    int_map: dict[int, float] = {}
    key_to_orig: dict[int, float] = {}
    for p in percentile_list:
        k = int(round(float(p.percentile) * 100))
        int_map[k] = float(p.value)
        key_to_orig[k] = float(p.percentile)

    widened = widen_numeric_percentiles(
        int_map,
        inflation_factor=inflation_factor,
        lower_bound=_to_numeric_bound(getattr(question, "lower_bound", None)),
        upper_bound=_to_numeric_bound(getattr(question, "upper_bound", None)),
        open_lower_bound=bool(getattr(question, "open_lower_bound", True)),
        open_upper_bound=bool(getattr(question, "open_upper_bound", True)),
    )

    cls = type(percentile_list[0])
    return [
        cls(percentile=key_to_orig[k], value=widened[k])
        for k in sorted(widened.keys())
    ]


def calibrate_multichoice_object(predicted_option_list: Any) -> Any:
    """Calibra un PredictedOptionList de forecasting_tools evitando ceros extremos."""
    options = getattr(predicted_option_list, "predicted_options", None)
    if not isinstance(options, list) or len(options) < 2:
        return predicted_option_list

    raw_map = {str(opt.option_name): float(opt.probability) for opt in options}
    calibrated = calibrate_multichoice_probs(raw_map)
    for opt in options:
        opt.probability = calibrated[str(opt.option_name)]
    return predicted_option_list
