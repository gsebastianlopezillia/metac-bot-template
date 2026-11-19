"""Tests unitarios para normalización de entorno Gemini/Metaculus y oracle_enrichment."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests

from bot_helpers import check_environment, load_and_normalize_env
from oracle_enrichment import (
    build_external_enrichment,
    extract_query_terms,
    fetch_noaa_enso_context,
    fetch_polymarket_prior,
    is_climate_question,
)


def test_load_and_normalize_env_maps_wonka_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Debe mapear METACULUS_BOT_1_API_KEY y GEMINI_METACULUS_BOT_1_API_KEY sin fallar."""
    for key in (
        "METACULUS_TOKEN",
        "METACULUS_BOT_1_API_KEY",
        "GEMINI_API_KEY",
        "GEMINI_METACULUS_BOT_1_API_KEY",
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)

    env_file = tmp_path / ".env"
    env_file.write_text(
        "METACULUS_BOT_1_API_KEY=test_metac_token\n"
        "GEMINI_METACULUS_BOT_1_API_KEY=test_gemini_key\n",
        encoding="utf-8",
    )

    load_and_normalize_env(repo_dir=tmp_path)

    assert os.environ["METACULUS_TOKEN"] == "test_metac_token"
    assert os.environ["GEMINI_API_KEY"] == "test_gemini_key"
    # No debe lanzar SystemExit
    check_environment(strict=True)


def test_extract_query_terms_and_climate_detection() -> None:
    terms = extract_query_terms("Will NOAA declare a Super El Nino in 2026?")
    assert "noaa" in terms
    assert "nino" in terms
    assert is_climate_question("Will NOAA declare a Super El Nino in 2026?") is True
    assert is_climate_question("Will Bitcoin exceed $150,000 in 2026?") is False


def test_fetch_polymarket_prior_matches_relevant_event_only() -> None:
    mock_session = MagicMock(spec=requests.Session)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "events": [
            {
                "title": "Will Federal Reserve cut interest rates in December?",
                "closed": False,
                "volume": 2500000,
                "markets": [
                    {"groupItemTitle": "25 bps cut", "outcomePrices": '["0.62", "0.38"]', "closed": False}
                ],
            },
            {
                "title": "Unrelated sports match",
                "closed": False,
                "volume": 50000,
                "markets": [
                    {"groupItemTitle": "Team A", "outcomePrices": '["0.50", "0.50"]', "closed": False}
                ],
            },
        ]
    }
    mock_session.get.return_value = mock_resp

    text = fetch_polymarket_prior(
        "Will the Federal Reserve cut interest rates in December?",
        session=mock_session,
    )
    assert "Federal Reserve cut interest rates" in text
    assert "62.0%" in text
    assert "Unrelated sports match" not in text


def test_fetch_noaa_enso_and_timeout_resilience() -> None:
    mock_session = MagicMock(spec=requests.Session)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = (
        " 14OCT2026     26.2 5.2     29.2 4.2     29.9 3.0     29.8 1.1\n"
        " 11NOV2026     27.1 5.4     29.4 4.3     29.9 3.2     29.9 1.3\n"
    )
    mock_session.get.return_value = mock_resp

    enso_text = fetch_noaa_enso_context(session=mock_session)
    assert "11NOV2026" in enso_text
    assert "+3.2°C" in enso_text

    # En caso de timeout, nunca debe lanzar excepción
    failing_session = MagicMock(spec=requests.Session)
    failing_session.get.side_effect = requests.Timeout("simulated timeout")
    assert build_external_enrichment("Will El Nino intensify?", session=failing_session) == ""
