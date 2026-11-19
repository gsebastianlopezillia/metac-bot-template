"""Tests unitarios del router de categorías y estratificación del dataset de benchmark."""

from __future__ import annotations

import pytest

from benchmark_dataset import stratify_records
from category_router import (
    OFFICIAL_CATEGORY_TO_PLAYBOOK,
    classify_question_data,
    normalize_category_name,
)


def test_normalize_strips_soft_hyphen_in_cryptocurrencies() -> None:
    """Metaculus devuelve 'Crypto\\xadcurrencies' con soft-hyphen invisible; debe normalizarlo."""
    raw = "Crypto\u00adcurrencies"
    assert normalize_category_name(raw) == "cryptocurrencies"
    route = classify_question_data(
        "Will Bitcoin reach $200k?",
        api_json={"projects": {"category": [{"name": raw}]}},
        question_type="BinaryQuestion",
    )
    assert route.playbook == "crypto"
    assert route.matched_by == "official_category"


@pytest.mark.parametrize(
    ("official_name", "expected_playbook"),
    [
        ("Economy & Business", "economy_business"),
        ("Elections", "politics_geo_law"),
        ("Geopolitics", "politics_geo_law"),
        ("Politics", "politics_geo_law"),
        ("Law", "politics_geo_law"),
        ("Nuclear Technology & Risks", "politics_geo_law"),
        ("Health & Pandemics", "health_science"),
        ("Natural Sciences", "health_science"),
        ("Sports & Entertainment", "sports_entertainment"),
        ("Environment & Climate", "climate_environment"),
        ("Cryptocurrencies", "crypto"),
        ("Artificial Intelligence", "ai_tech_space"),
        ("Technology", "ai_tech_space"),
        ("Computing and Math", "ai_tech_space"),
        ("Space", "ai_tech_space"),
    ],
)
def test_all_15_official_metaculus_categories_are_mapped(
    official_name: str, expected_playbook: str
) -> None:
    route = classify_question_data(
        "Sample question",
        api_json={"projects": {"category": [{"name": official_name}]}},
        question_type="BinaryQuestion",
    )
    assert route.playbook == expected_playbook
    assert route.matched_by == "official_category"


def test_keyword_fallback_and_general_fallback_cover_untagged_questions() -> None:
    r_law = classify_question_data("Will the Supreme Court overturn the tariff before 2027?")
    assert r_law.playbook == "politics_geo_law"
    assert r_law.matched_by == "keyword_fallback"

    r_climate = classify_question_data("Will NOAA report El Niño anomaly above +2.5C?")
    assert r_climate.playbook == "climate_environment"
    assert r_climate.matched_by == "keyword_fallback"

    r_weird = classify_question_data("Will an obscure local festival break attendance records?")
    assert r_weird.playbook == "general_fallback"
    assert r_weird.matched_by == "default_fallback"


def test_stratify_records_balances_playbooks_and_excludes_ambiguous() -> None:
    records = [
        {"question_id": 1, "playbook": "economy_business", "question_type": "BinaryQuestion", "resolution": "yes"},
        {"question_id": 2, "playbook": "economy_business", "question_type": "NumericQuestion", "resolution": 4.2},
        {"question_id": 3, "playbook": "economy_business", "question_type": "MultipleChoiceQuestion", "resolution": "A"},
        {"question_id": 4, "playbook": "economy_business", "question_type": "BinaryQuestion", "resolution": "ambiguous"},
        {"question_id": 5, "playbook": "crypto", "question_type": "BinaryQuestion", "resolution": "no"},
    ]
    selected = stratify_records(records, per_playbook=2)
    ids = {r["question_id"] for r in selected}
    # Ambiguous excluida; economy_business toma 1 Binary + 1 Numeric (round-robin); crypto toma la suya
    assert 4 not in ids
    assert ids == {1, 2, 5}
    assert len(OFFICIAL_CATEGORY_TO_PLAYBOOK) >= 15
