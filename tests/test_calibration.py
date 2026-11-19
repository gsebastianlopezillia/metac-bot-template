"""Tests unitarios para el módulo de calibración matemática."""

import math
import pytest

from calibration import (
    apply_status_quo_bias,
    blend_with_market_prior,
    calibrate_binary_forecast,
    calibrate_multichoice_probs,
    clamp_binary_tails,
    log_odds_to_prob,
    prob_to_log_odds,
    shrink_binary_probability,
    widen_numeric_percentiles,
)


def test_log_odds_roundtrip():
    for p in (0.05, 0.25, 0.5, 0.73, 0.95):
        assert math.isclose(log_odds_to_prob(prob_to_log_odds(p)), p, abs_tol=1e-9)


def test_shrink_pulls_extremes_toward_center():
    assert shrink_binary_probability(0.90, alpha=0.80) < 0.90
    assert shrink_binary_probability(0.10, alpha=0.80) > 0.10
    assert math.isclose(shrink_binary_probability(0.50, alpha=0.80), 0.50, abs_tol=1e-9)


def test_blend_with_market_prior_respects_log_odds_average():
    # Si modelo=0.80 y mercado=0.20 con peso 0.50, el punto medio exacto en log-odds es 0.50
    blended = blend_with_market_prior(0.80, 0.20, market_weight=0.50)
    assert math.isclose(blended, 0.50, abs_tol=1e-6)


def test_status_quo_bias_only_applies_on_short_horizons_and_non_settled_probs():
    # Horizonte corto (2 días) reduce una probabilidad incierta (0.60)
    short = apply_status_quo_bias(0.60, days_remaining=2, status_quo_is_no=True)
    assert short < 0.60

    # Probabilidad mayoritaria (>0.65) no se empuja ciegamente hacia NO (continuidad)
    assert apply_status_quo_bias(0.80, days_remaining=2, status_quo_is_no=True) == 0.80

    # Horizonte largo (30 días) no altera
    assert apply_status_quo_bias(0.60, days_remaining=30) == 0.60

    # Hecho casi consumado (0.95) no se toca por status quo
    assert apply_status_quo_bias(0.95, days_remaining=2) == 0.95


def test_calibrate_binary_pipeline_clamps_and_blends():
    res = calibrate_binary_forecast(
        0.60,
        market_prior=0.45,
        market_weight=0.50,
        shrink_alpha=0.80,
        days_remaining=3,
        apply_status_quo=True,
    )
    assert 0.40 < res.calibrated_probability < 0.60
    assert len(res.applied_steps) == 3

    # Sin mercado, una probabilidad extrema 0.999 queda acotada por clamp
    extreme = calibrate_binary_forecast(0.999, low_clamp=0.03, high_clamp=0.97)
    assert extreme.calibrated_probability <= 0.97
    assert clamp_binary_tails(0.001, low=0.03, high=0.97) == 0.03


def test_widen_numeric_percentiles_expands_tails_and_respects_closed_bounds():
    raw = {10: 40.0, 20: 45.0, 40: 48.0, 50: 50.0, 60: 52.0, 80: 55.0, 90: 60.0}
    widened = widen_numeric_percentiles(
        raw,
        inflation_factor=1.30,
        lower_bound=0.0,
        upper_bound=100.0,
        open_lower_bound=False,
        open_upper_bound=False,
    )
    assert widened[50] == 50.0
    assert widened[10] < raw[10]
    assert widened[90] > raw[90]
    # Verificar monotonicidad estricta
    ordered = [widened[k] for k in sorted(widened)]
    assert all(ordered[i] < ordered[i + 1] for i in range(len(ordered) - 1))


def test_widen_numeric_percentiles_works_without_explicit_50_and_validates():
    # Formato estándar de FallTemplateBot2026: 10, 20, 40, 60, 80, 90 (sin 50)
    raw = {10: 40.0, 20: 45.0, 40: 48.0, 60: 52.0, 80: 55.0, 90: 60.0}
    widened = widen_numeric_percentiles(raw, inflation_factor=1.25)
    assert widened[10] < raw[10]
    assert widened[90] > raw[90]

    with pytest.raises(ValueError):
        widen_numeric_percentiles({50: 5.0})
    with pytest.raises(ValueError):
        widen_numeric_percentiles({10: 1.0, 90: 9.0}, inflation_factor=0.5)


def test_calibrate_multichoice_probs_prevents_zeros_and_sums_to_one():
    raw = {"A": 0.98, "B": 0.02, "C": 0.0}
    cal = calibrate_multichoice_probs(raw, uniform_shrink=0.08, min_floor=0.015)
    assert math.isclose(sum(cal.values()), 1.0, abs_tol=1e-4)
    assert cal["C"] >= 0.015
    assert cal["A"] < 0.98


def test_dispatcher_core_rules_include_historical_memory_bias():
    from playbooks.dispatcher import CORE_FORECASTING_RULES

    assert "Viejo Memorioso" in CORE_FORECASTING_RULES
    assert "Tier-1 Evidence Supremacy" in CORE_FORECASTING_RULES
