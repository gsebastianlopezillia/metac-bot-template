"""Calibración matemática determinista (0 tokens) para pronósticos de Metaculus.

Corrige sesgos sistemáticos de los LLMs (sobreconfianza en binarias, intervalos
estrechos en numéricas) e inyecta priors cuantitativos (anclaje de mercado y
fricción de status quo) en espacio log-odds.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


_EPS = 1e-6


@dataclass(frozen=True)
class BinaryCalibrationResult:
    """Resultado trazable de la calibración de una probabilidad binaria."""

    raw_probability: float
    calibrated_probability: float
    market_prior: float | None
    applied_steps: tuple[str, ...]


def prob_to_log_odds(probability: float) -> float:
    """Convierte una probabilidad en (0, 1) a log-odds."""
    clipped = min(max(probability, _EPS), 1.0 - _EPS)
    return math.log(clipped / (1.0 - clipped))


def log_odds_to_prob(log_odds: float) -> float:
    """Convierte log-odds a probabilidad en (0, 1)."""
    if log_odds >= 0:
        z = math.exp(-log_odds)
        return 1.0 / (1.0 + z)
    z = math.exp(log_odds)
    return z / (1.0 + z)


def shrink_binary_probability(
    probability: float,
    alpha: float = 0.80,
    anchor: float = 0.50,
) -> float:
    """Comprime extremos en log-odds hacia un ancla para mitigar sobreconfianza.

    alpha=1.0 no altera la probabilidad; alpha=0.0 la lleva por completo al ancla.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha debe estar en [0.0, 1.0]")
    lo_p = prob_to_log_odds(probability)
    lo_anchor = prob_to_log_odds(anchor)
    return log_odds_to_prob(lo_anchor + alpha * (lo_p - lo_anchor))


def blend_with_market_prior(
    model_probability: float,
    market_probability: float,
    market_weight: float = 0.50,
) -> float:
    """Combina el pronóstico del LLM con un precio de mercado en espacio log-odds."""
    if not 0.0 <= market_weight <= 1.0:
        raise ValueError("market_weight debe estar en [0.0, 1.0]")
    lo_model = prob_to_log_odds(model_probability)
    lo_market = prob_to_log_odds(market_probability)
    blended_lo = (1.0 - market_weight) * lo_model + market_weight * lo_market
    return log_odds_to_prob(blended_lo)


def apply_status_quo_bias(
    probability: float,
    days_remaining: int | None,
    status_quo_is_no: bool = True,
    max_shift_log_odds: float = 0.35,
) -> float:
    """Aplica fricción institucional cuando queda poco tiempo para que algo cambie.

    Solo actúa si el horizonte es corto (<= 14 días), la probabilidad no está ya
    en zona de hecho consumado y —cuando status_quo_is_no=True— no contradice un
    estado actual ya mayoritario (> 0.65, típico de preguntas de continuidad).
    """
    if days_remaining is None or days_remaining < 0 or days_remaining > 14:
        return probability
    if probability >= 0.92 or probability <= 0.08:
        return probability
    if status_quo_is_no and probability > 0.65:
        return probability

    # Más fuerte cuanto menos días faltan (lineal de 14 días a 0 días)
    horizon_factor = (14.0 - float(days_remaining)) / 14.0
    shift = max_shift_log_odds * horizon_factor
    lo = prob_to_log_odds(probability)
    adjusted_lo = lo - shift if status_quo_is_no else lo + shift
    return log_odds_to_prob(adjusted_lo)


def clamp_binary_tails(
    probability: float,
    low: float = 0.03,
    high: float = 0.97,
) -> float:
    """Acota colas extremas para proteger el Log Score contra sorpresas."""
    if not 0.0 < low < high < 1.0:
        raise ValueError("Se requiere 0 < low < high < 1")
    return min(max(probability, low), high)


def calibrate_binary_forecast(
    raw_probability: float,
    *,
    market_prior: float | None = None,
    market_weight: float = 0.50,
    shrink_alpha: float = 0.82,
    days_remaining: int | None = None,
    apply_status_quo: bool = False,
    low_clamp: float = 0.03,
    high_clamp: float = 0.97,
) -> BinaryCalibrationResult:
    """Pipeline unificado de calibración para preguntas binarias."""
    steps: list[str] = []
    p = min(max(raw_probability, _EPS), 1.0 - _EPS)

    # 1. Mitigar sobreconfianza del LLM
    p_shrunk = shrink_binary_probability(p, alpha=shrink_alpha, anchor=0.50)
    steps.append(f"shrink(alpha={shrink_alpha:.2f}): {p:.3f}->{p_shrunk:.3f}")
    p = p_shrunk

    # 2. Fricción institucional en plazos cortos si corresponde al subtipo
    if apply_status_quo and days_remaining is not None:
        p_sq = apply_status_quo_bias(p, days_remaining=days_remaining)
        if abs(p_sq - p) > 1e-4:
            steps.append(f"status_quo(days={days_remaining}): {p:.3f}->{p_sq:.3f}")
            p = p_sq

    # 3. Anclaje con mercado externo si existe match fuerte
    if market_prior is not None and 0.01 <= market_prior <= 0.99:
        p_blend = blend_with_market_prior(p, market_prior, market_weight=market_weight)
        steps.append(
            f"market_blend(prior={market_prior:.3f}, w={market_weight:.2f}): "
            f"{p:.3f}->{p_blend:.3f}"
        )
        p = p_blend

    # 4. Protección de colas para Log Score
    p_clamped = clamp_binary_tails(p, low=low_clamp, high=high_clamp)
    if abs(p_clamped - p) > 1e-6:
        steps.append(f"clamp([{low_clamp:.2f},{high_clamp:.2f}]): {p:.3f}->{p_clamped:.3f}")
        p = p_clamped

    return BinaryCalibrationResult(
        raw_probability=raw_probability,
        calibrated_probability=round(p, 4),
        market_prior=market_prior,
        applied_steps=tuple(steps),
    )


def widen_numeric_percentiles(
    percentiles: dict[int, float],
    *,
    inflation_factor: float = 1.25,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    open_lower_bound: bool = True,
    open_upper_bound: bool = True,
) -> dict[int, float]:
    """Ensancha percentiles alrededor de P50 para corregir intervalos estrechos.

    Preserva monotonicidad estricta y respeta los límites cerrados de Metaculus.
    """
    if inflation_factor < 1.0:
        raise ValueError("inflation_factor debe ser >= 1.0")
    if len(percentiles) < 2:
        raise ValueError("Se requieren al menos 2 percentiles para ensanchar")

    keys = sorted(percentiles.keys())
    if 50 in percentiles:
        median = percentiles[50]
    elif 40 in percentiles and 60 in percentiles:
        median = (percentiles[40] + percentiles[60]) / 2.0
    else:
        mid = len(keys) // 2
        median = percentiles[keys[mid]]
    widened: dict[int, float] = {}

    for k in keys:
        val = percentiles[k]
        if k == 50:
            new_val = median
        else:
            new_val = median + (val - median) * inflation_factor
        if lower_bound is not None and not open_lower_bound:
            new_val = max(new_val, lower_bound)
        if upper_bound is not None and not open_upper_bound:
            new_val = min(new_val, upper_bound)
        widened[k] = new_val

    # Garantizar orden estrictamente creciente (requisito de Metaculus CDF)
    for idx in range(1, len(keys)):
        prev_k = keys[idx - 1]
        curr_k = keys[idx]
        if widened[curr_k] <= widened[prev_k]:
            step = max(abs(widened[prev_k]) * 1e-4, 1e-4)
            widened[curr_k] = widened[prev_k] + step

    return {k: round(widened[k], 6) for k in keys}


def calibrate_multichoice_probs(
    probs: dict[str, float],
    *,
    uniform_shrink: float = 0.08,
    min_floor: float = 0.015,
) -> dict[str, float]:
    """Calibra probabilidades de opción múltiple para proteger el Log Score.

    Aplica un suavizado hacia la distribución uniforme (frenando sobreconfianza),
    impone un piso mínimo > 0 por opción (evita muerte súbita en Log Score) y
    renormaliza exactamente a suma 1.0.
    """
    if not probs:
        return {}
    if not 0.0 <= uniform_shrink <= 0.5:
        raise ValueError("uniform_shrink debe estar en [0.0, 0.5]")

    n = len(probs)
    if n == 1:
        only_key = next(iter(probs))
        return {only_key: 1.0}

    cleaned = {k: max(0.0, float(v)) for k, v in probs.items()}
    total = sum(cleaned.values())
    base = {k: (v / total if total > 0 else 1.0 / n) for k, v in cleaned.items()}

    # Suavizado hacia uniforme (1/n)
    uniform = 1.0 / n
    shrunk = {
        k: (1.0 - uniform_shrink) * v + uniform_shrink * uniform
        for k, v in base.items()
    }

    # Piso mínimo seguro (si hay muchas opciones, acotar el piso a 1/(2*n))
    effective_floor = min(min_floor, 0.5 / n)
    floored = {k: max(v, effective_floor) for k, v in shrunk.items()}
    norm_total = sum(floored.values())
    normalized = {k: round(v / norm_total, 4) for k, v in floored.items()}

    # Ajustar redondeo en la opción mayor para garantizar suma exacta 1.0000
    diff = round(1.0 - sum(normalized.values()), 4)
    if abs(diff) > 0:
        max_k = max(normalized, key=lambda k: normalized[k])
        normalized[max_k] = round(normalized[max_k] + diff, 4)
    return normalized
