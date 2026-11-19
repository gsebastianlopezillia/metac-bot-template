"""Generador de dataset estratificado de benchmark (0 tokens de LLM).

Descarga preguntas resueltas reales de Metaculus, las clasifica con `category_router.py`
y congela un JSON balanceado por playbook y formato (Binary/Numeric/MC) para auditar
nuestro bot contra el template vanilla y la comunidad antes de publicar en torneo.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from bot_helpers import load_and_normalize_env, silence_noisy_dependencies
from category_router import PLAYBOOK_PRIORITY, classify_metaculus_question

DEFAULT_OUTPUT_PATH = Path("benchmarks/dataset_v1.json")


def _extract_resolution(question: Any) -> Any:
    raw = getattr(question, "api_json", None) or {}
    q_obj = raw.get("question") if isinstance(raw, dict) else None
    if isinstance(q_obj, dict) and q_obj.get("resolution") is not None:
        return q_obj.get("resolution")
    return getattr(question, "resolution", None)


def _extract_community_prediction(question: Any) -> Any:
    raw = getattr(question, "api_json", None) or {}
    q_obj = raw.get("question") if isinstance(raw, dict) else None
    if not isinstance(q_obj, dict):
        return None
    aggs = q_obj.get("aggregations") or {}
    recency = aggs.get("recency_weighted") if isinstance(aggs, dict) else None
    latest = recency.get("latest") if isinstance(recency, dict) else None
    if isinstance(latest, dict):
        return latest.get("centers") or latest.get("forecast_values")
    return None


def serialize_benchmark_question(question: Any) -> dict[str, Any]:
    """Convierte una pregunta resuelta del SDK en registro limpio y serializable."""
    route = classify_metaculus_question(question)
    return {
        "post_id": getattr(question, "id_of_post", None),
        "question_id": getattr(question, "id_of_question", None),
        "url": str(getattr(question, "page_url", "") or ""),
        "question_text": str(getattr(question, "question_text", "") or ""),
        "question_type": route.question_type,
        "playbook": route.playbook,
        "official_categories": list(route.official_categories),
        "matched_by": route.matched_by,
        "resolution": _extract_resolution(question),
        "community_prediction": _extract_community_prediction(question),
    }


def stratify_records(
    records: list[dict[str, Any]],
    per_playbook: int = 6,
) -> list[dict[str, Any]]:
    """Selecciona hasta `per_playbook` preguntas por dominio balanceando Binary/Numeric/MC."""
    by_playbook: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_ids: set[Any] = set()

    for rec in records:
        qid = rec.get("question_id") or rec.get("url")
        if qid in seen_ids or rec.get("resolution") in ("ambiguous", "annulled"):
            continue
        seen_ids.add(qid)
        by_playbook[rec["playbook"]].append(rec)

    selected: list[dict[str, Any]] = []
    for pb in PLAYBOOK_PRIORITY:
        pool = by_playbook.get(pb, [])
        if not pool:
            continue
        # Rotación round-robin por tipo de pregunta para no quedar solo con Binary
        by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in pool:
            by_type[item["question_type"]].append(item)
        picked: list[dict[str, Any]] = []
        type_order = ["BinaryQuestion", "NumericQuestion", "MultipleChoiceQuestion", "DateQuestion"]
        while len(picked) < per_playbook:
            added = False
            for qtype in type_order:
                bucket = by_type.get(qtype)
                if bucket:
                    picked.append(bucket.pop(0))
                    added = True
                    if len(picked) >= per_playbook:
                        break
            if not added:
                break
        selected.extend(picked)

    return selected


async def build_dataset(pool_size: int = 200, per_playbook: int = 6, output: Path = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    """Descarga preguntas resueltas de Metaculus, clasifica y guarda el dataset congelado."""
    load_and_normalize_env()
    silence_noisy_dependencies()
    from forecasting_tools import ApiFilter, MetaculusClient

    client = MetaculusClient()
    questions = await client.get_questions_matching_filter(
        ApiFilter(
            allowed_statuses=["resolved"],
            allowed_types=["binary", "numeric", "multiple_choice", "date"],
        ),
        num_questions=pool_size,
    )

    records = [serialize_benchmark_question(q) for q in questions]
    stratified = stratify_records(records, per_playbook=per_playbook)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pool_scanned": len(records),
        "selected_count": len(stratified),
        "by_playbook": dict(Counter(r["playbook"] for r in stratified)),
        "by_question_type": dict(Counter(r["question_type"] for r in stratified)),
        "by_match_source": dict(Counter(r["matched_by"] for r in stratified)),
        "questions": stratified,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Construye dataset estratificado de Metaculus (0 tokens LLM).")
    parser.add_argument("--pool-size", type=int, default=200, help="Cantidad de preguntas resueltas a escanear.")
    parser.add_argument("--per-playbook", type=int, default=6, help="Máximo de preguntas por playbook.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Archivo JSON de salida.")
    args = parser.parse_args()

    summary = asyncio.run(build_dataset(args.pool_size, args.per_playbook, args.output))
    print(f"Dataset guardado en {args.output.resolve()}")
    print(f"Escaneadas: {summary['pool_scanned']} | Seleccionadas: {summary['selected_count']}")
    print("Por playbook:", summary["by_playbook"])
    print("Por formato:", summary["by_question_type"])
    print("Por método de clasificación:", summary["by_match_source"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
