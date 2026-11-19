"""Tests unitarios offline para el playbook sports_entertainment."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

from playbooks.sports_entertainment import (
    build_sports_entertainment_context,
    detect_espn_league,
    detect_sports_subtype,
    fetch_espn_matchup_context,
)


def _mock_response(status_code: int = 200, *, json_data=None, text: str = ""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    if json_data is not None:
        resp.json.return_value = json_data
    return resp


def test_detect_sports_subtype_and_espn_league():
    assert detect_sports_subtype("Will Detroit Red Wings beat Florida Panthers?") == "match_outcome"
    assert detect_sports_subtype("Will Czechia reach the semi-finals of IIHF World Championship?") == "tournament_advancement"
    assert detect_sports_subtype("Will the 2026 American League MVP Award be unanimous?") == "awards_records"
    assert detect_sports_subtype("Will Dune 3 top the weekend box office?") == "entertainment_media"

    assert detect_espn_league("Detroit Red Wings vs Florida Panthers") == ("hockey", "nhl")
    assert detect_espn_league("Italy vs France UEFA Nations League") == ("soccer", "uefa.nations")
    assert detect_espn_league("Bundesliga Matchday 6") == ("soccer", "ger.1")


def test_fetch_espn_matchup_context_extracts_records_and_odds():
    session = MagicMock()
    session.get.return_value = _mock_response(
        200,
        json_data={
            "events": [
                {
                    "name": "Detroit Red Wings at Florida Panthers",
                    "date": "2026-11-12T23:00Z",
                    "competitions": [
                        {
                            "competitors": [
                                {"team": {"abbreviation": "DET"}, "records": [{"summary": "9-6-1"}]},
                                {"team": {"abbreviation": "FLA"}, "records": [{"summary": "11-4-2"}]},
                            ],
                            "odds": [{"details": "FLA -155", "overUnder": 6.0}],
                        }
                    ],
                }
            ]
        },
    )
    lines = fetch_espn_matchup_context(
        "Will the Detroit Red Wings beat the Florida Panthers?",
        session=session,
    )
    assert len(lines) == 1
    assert "FLA -155" in lines[0]
    assert "DET (9-6-1)" in lines[0]


def test_build_sports_context_assembles_tiers_and_unanimity_bias():
    session = MagicMock()
    session.get.return_value = _mock_response(200, json_data=[], text="<rss><channel></channel></rss>")

    ctx = build_sports_entertainment_context(
        "Will the winner of the 2026 American League MVP Award be a unanimous selection?",
        session=session,
        now=datetime(2026, 11, 10, tzinfo=timezone.utc),
    )
    assert ctx.playbook == "sports_entertainment"
    assert ctx.subtype == "awards_records"
    assert ctx.apply_status_quo_bias is True
    assert "Unanimous" in ctx.prompt_addendum
