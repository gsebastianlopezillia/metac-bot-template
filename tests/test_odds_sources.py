"""Tests unitarios offline (0 créditos reales) para The Odds API y vig-free math."""

import json
import math
from unittest.mock import MagicMock

from playbooks.odds_sources import (
    american_odds_to_implied_prob,
    decimal_odds_to_implied_prob,
    fetch_the_odds_api_consensus,
    remove_vig,
)


def test_vig_free_conversion_normalizes_bookmaker_overround():
    p_fav = american_odds_to_implied_prob(-155)
    p_dog = american_odds_to_implied_prob(135)
    assert p_fav + p_dog > 1.0  # incluye margen de la casa (vig)

    clean = remove_vig({"FLA": p_fav, "DET": p_dog})
    assert math.isclose(sum(clean.values()), 1.0, abs_tol=1e-3)
    assert 0.58 < clean["FLA"] < 0.60

    assert math.isclose(decimal_odds_to_implied_prob(2.0), 0.50, abs_tol=1e-9)


def test_the_odds_api_consensus_uses_cache_and_extracts_subject_prob(monkeypatch, tmp_path):
    monkeypatch.setenv("THE_ODDS_API_KEY", "dummy_test_key")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"x-requests-remaining": "495"}
    mock_resp.json.return_value = [
        {
            "home_team": "Florida Panthers",
            "away_team": "Detroit Red Wings",
            "commence_time": "2026-11-20T00:00:00Z",
            "bookmakers": [
                {
                    "key": "draftkings",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Florida Panthers", "price": 1.65},
                                {"name": "Detroit Red Wings", "price": 2.30},
                            ],
                        }
                    ],
                },
                {
                    "key": "fanduel",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Florida Panthers", "price": 1.62},
                                {"name": "Detroit Red Wings", "price": 2.35},
                            ],
                        }
                    ],
                },
            ],
        }
    ]
    session = MagicMock()
    session.get.return_value = mock_resp

    # Primera llamada consulta la API y guarda caché
    c1 = fetch_the_odds_api_consensus(
        "Will the Detroit Red Wings beat the Florida Panthers?",
        days_remaining=2,
        session=session,
        cache_dir=tmp_path,
    )
    assert c1 is not None
    assert c1.bookmaker_count == 2
    assert c1.target_team_prob is not None and 0.39 < c1.target_team_prob < 0.43
    assert session.get.call_count == 1

    # Segunda llamada dentro de 6h reutiliza caché (0 créditos extra)
    c2 = fetch_the_odds_api_consensus(
        "Will the Detroit Red Wings beat the Florida Panthers?",
        days_remaining=2,
        session=session,
        cache_dir=tmp_path,
    )
    assert c2 is not None
    assert session.get.call_count == 1


def test_the_odds_api_respects_credit_reserve_and_horizon_guard(monkeypatch, tmp_path):
    monkeypatch.setenv("THE_ODDS_API_KEY", "dummy_test_key")
    session = MagicMock()

    # Horizonte > 14 días no gasta crédito
    assert (
        fetch_the_odds_api_consensus(
            "Will the Detroit Red Wings beat the Florida Panthers?",
            days_remaining=30,
            session=session,
            cache_dir=tmp_path,
        )
        is None
    )
    assert session.get.call_count == 0

    # Si quedan menos de 25 créditos, entra en modo reserva sin llamar a la API
    (tmp_path / "_quota_state.json").write_text(json.dumps({"remaining": 10}), encoding="utf-8")
    assert (
        fetch_the_odds_api_consensus(
            "Will the Detroit Red Wings beat the Florida Panthers?",
            days_remaining=1,
            session=session,
            cache_dir=tmp_path,
        )
        is None
    )
    assert session.get.call_count == 0
