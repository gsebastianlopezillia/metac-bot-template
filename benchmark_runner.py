"""Laboratorio offline con caché local para afinar playbooks sin quemar tokens.

Modos:
- --dry-research (0 tokens LLM): corre fuentes curadas + clasificación de subtipo
  y guarda caché en benchmarks/cache/ para auditar qué contexto ve el modelo.
- --compare: compara Vanilla vs Playbook+Calibración en 1 llamada por variante
  usando gemini/gemini-3.5-flash-lite (o el modelo configurado en .env).
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import re
from typing import Any

from bot_helpers import build_llm_config, load_and_normalize_env
from calibration import calibrate_binary_forecast
from playbooks.ai_tech_space import build_ai_tech_space_context
from playbooks.climate_environment import build_climate_environment_context
from playbooks.crypto import build_crypto_context
from playbooks.economy_business import build_economy_business_context
from playbooks.health_science import build_health_science_context
from playbooks.politics_geo_law import PlaybookContext, build_politics_geo_law_context
from playbooks.sports_entertainment import build_sports_entertainment_context

DATASET_DEFAULT = Path("benchmarks/dataset_v1.json")
CACHE_DIR = Path("benchmarks/cache")
_PROB_RE = re.compile(r"PROBABILITY\s*:\s*(\d+(?:\.\d+)?)\s*%?", re.IGNORECASE)


def _load_questions(dataset_path: Path, playbook: str, max_questions: int) -> list[dict[str, Any]]:
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    items = [q for q in payload.get("questions", []) if q.get("playbook") == playbook]
    return items[:max_questions]


def _cache_path(question_id: int) -> Path:
    return CACHE_DIR / f"research_{question_id}.json"


def load_or_build_context(
    item: dict[str, Any],
    *,
    refresh_cache: bool = False,
) -> PlaybookContext:
    """Obtiene contexto del playbook desde caché local o fuentes abiertas."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(int(item["question_id"]))
    if path.exists() and not refresh_cache:
        data = json.loads(path.read_text(encoding="utf-8"))
        return PlaybookContext(**data)

    playbook_name = str(item.get("playbook") or "politics_geo_law")
    if playbook_name == "economy_business":
        ctx = build_economy_business_context(str(item["question_text"]))
    elif playbook_name == "health_science":
        ctx = build_health_science_context(str(item["question_text"]))
    elif playbook_name == "sports_entertainment":
        ctx = build_sports_entertainment_context(str(item["question_text"]))
    elif playbook_name == "climate_environment":
        ctx = build_climate_environment_context(str(item["question_text"]))
    elif playbook_name == "ai_tech_space":
        ctx = build_ai_tech_space_context(str(item["question_text"]))
    elif playbook_name == "crypto":
        ctx = build_crypto_context(str(item["question_text"]))
    else:
        ctx = build_politics_geo_law_context(
            str(item["question_text"]),
            official_categories=item.get("official_categories") or [],
        )
    path.write_text(json.dumps(ctx.__dict__, indent=2, ensure_ascii=False), encoding="utf-8")
    return ctx


def _extract_prob_from_text(text: str) -> float | None:
    match = _PROB_RE.search(text)
    if not match:
        return None
    val = float(match.group(1))
    if val > 1.0:
        val = val / 100.0
    if 0.0 <= val <= 1.0:
        return val
    return None


async def _run_compare_on_question(
    item: dict[str, Any],
    ctx: PlaybookContext,
    llm: Any,
) -> dict[str, Any]:
    q_text = str(item["question_text"])
    q_type = str(item["question_type"])

    if q_type != "BinaryQuestion":
        return {
            "question_id": item["question_id"],
            "question_type": q_type,
            "subtype": ctx.subtype,
            "note": "Non-binary: inspect evidence cache (binary A/B harness active)",
        }

    vanilla_prompt = (
        "You are a superforecaster. Estimate the probability (0 to 100%) that "
        f"this question resolves YES:\n\n{q_text}\n\n"
        "Respond concisely in 3 bullet points and end with: PROBABILITY: XX%"
    )
    playbook_prompt = (
        "You are a superforecaster following a strict domain protocol.\n\n"
        f"Question: {q_text}\n\n"
        f"{ctx.prompt_addendum}\n\n"
        f"{ctx.evidence_text}\n\n"
        "Respond concisely (Outside View, Key Evidence, Pre-Mortem) and end with: PROBABILITY: XX%"
    )

    vanilla_raw = await llm.invoke(vanilla_prompt)
    playbook_raw = await llm.invoke(playbook_prompt)

    p_vanilla = _extract_prob_from_text(str(vanilla_raw))
    p_playbook_raw = _extract_prob_from_text(str(playbook_raw))
    cal = (
        calibrate_binary_forecast(
            p_playbook_raw,
            market_prior=ctx.market_prior,
            market_weight=ctx.market_weight,
            shrink_alpha=ctx.default_shrink_alpha,
            days_remaining=ctx.days_remaining,
            apply_status_quo=ctx.apply_status_quo_bias,
        )
        if p_playbook_raw is not None
        else None
    )

    return {
        "question_id": item["question_id"],
        "question_type": q_type,
        "subtype": ctx.subtype,
        "vanilla_p": p_vanilla,
        "playbook_raw_p": p_playbook_raw,
        "playbook_calibrated_p": cal.calibrated_probability if cal else None,
        "market_prior": ctx.market_prior,
        "calibration_steps": list(cal.applied_steps) if cal else [],
    }


async def run_benchmark(
    *,
    dataset_path: Path,
    playbook: str,
    max_questions: int,
    dry_research: bool,
    refresh_cache: bool,
) -> None:
    items = _load_questions(dataset_path, playbook, max_questions)
    print(f"Loaded {len(items)} questions for playbook='{playbook}' (dry_research={dry_research})")

    contexts: list[tuple[dict[str, Any], PlaybookContext]] = []
    for item in items:
        ctx = load_or_build_context(item, refresh_cache=refresh_cache)
        contexts.append((item, ctx))
        print(
            f"- [{item['question_id']}] ({item['question_type']}) subtype={ctx.subtype} "
            f"| market_prior={ctx.market_prior} ({ctx.market_prior_source}) "
            f"| evidence_chars={len(ctx.evidence_text)}"
        )

    if dry_research:
        print(f"\nCache saved under {CACHE_DIR.resolve()} (0 LLM tokens used).")
        return

    load_and_normalize_env()
    from forecasting_tools import GeneralLlm

    llm_cfg = build_llm_config()
    llm: GeneralLlm = llm_cfg["default"]

    results = []
    for item, ctx in contexts:
        res = await _run_compare_on_question(item, ctx, llm)
        results.append(res)
        print(json.dumps(res, ensure_ascii=False))

    out_path = CACHE_DIR / f"compare_{playbook}.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nComparison report saved to {out_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline playbook benchmark runner")
    parser.add_argument("--dataset", type=Path, default=DATASET_DEFAULT)
    parser.add_argument("--playbook", type=str, default="politics_geo_law")
    parser.add_argument("--max-questions", type=int, default=6)
    parser.add_argument("--dry-research", action="store_true", help="0 LLM tokens; only build/cache research")
    parser.add_argument("--refresh-cache", action="store_true", help="Ignore existing research cache")
    args = parser.parse_args()

    asyncio.run(
        run_benchmark(
            dataset_path=args.dataset,
            playbook=args.playbook,
            max_questions=args.max_questions,
            dry_research=args.dry_research,
            refresh_cache=args.refresh_cache,
        )
    )


if __name__ == "__main__":
    main()
