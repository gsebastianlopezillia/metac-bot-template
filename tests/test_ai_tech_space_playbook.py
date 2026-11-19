"""Tests unitarios offline para ai_tech_space (SpaceX/BO/Virgin/JAXA/ISRO/China + GitHub)."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

from playbooks.ai_tech_space import (
    build_ai_tech_space_context,
    detect_tech_space_subtype,
)
from playbooks.tech_space_sources import (
    detect_space_provider,
    extract_github_repos,
    fetch_github_repo_summary,
    fetch_launch_library_summary,
)


def _mock_response(status_code: int = 200, *, json_data=None, text: str = ""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    if json_data is not None:
        resp.json.return_value = json_data
    return resp


def test_detect_all_six_space_providers_and_subtypes():
    assert detect_space_provider("How many Falcon 9 launches will SpaceX have?") == "SpaceX"
    assert detect_space_provider("Will Blue Origin launch New Glenn this month?") == "Blue Origin"
    assert detect_space_provider("Will Virgin Galactic fly commercial passengers?") == "Virgin"
    assert detect_space_provider("Will Japan's JAXA H3 rocket reach orbit?") == "JAXA"
    assert detect_space_provider("Will India's ISRO launch Gaganyaan uncrewed?") == "ISRO"
    assert detect_space_provider("How many Long March rockets will China launch?") == "China"

    assert detect_tech_space_subtype("How many Falcon 9 launches will SpaceX have?") == "space_launches"
    assert detect_tech_space_subtype("Will the openai/math repository report > 300 results?") == "github_open_source"
    assert detect_tech_space_subtype("Will Anthropic release Claude Haiku 5.5?") == "ai_models_releases"
    assert detect_tech_space_subtype("How many license plate readers on DeFlock.org?") == "tech_dashboards_general"


def test_extract_github_repos_ignores_ratios():
    assert extract_github_repos("Will openai/math report >300 results (y/y)?") == ["openai/math"]
    assert extract_github_repos("Oil demand change y/y in mb/d") == []


def test_fetch_launch_library_and_github_repo_summary():
    session = MagicMock()
    session.get.return_value = _mock_response(
        200,
        json_data={
            "results": [
                {
                    "net": "2026-11-15T12:00:00Z",
                    "name": "LVM3 | Gaganyaan Test",
                    "launch_service_provider": {"name": "ISRO"},
                    "status": {"abbrev": "Success"},
                }
            ]
        },
    )
    prev_l, next_l = fetch_launch_library_summary("Will India's ISRO launch LVM3?", session=session)
    assert len(prev_l) == 1
    assert prev_l[0].provider == "ISRO"
    assert len(next_l) == 1

    gh_session = MagicMock()
    gh_session.get.side_effect = [
        _mock_response(200, json_data={"stargazers_count": 420, "pushed_at": "2026-11-18T10:00:00Z", "description": "Lean proofs"}),
        _mock_response(200, json_data=[{"commit": {"message": "Add theorem 305", "author": {"date": "2026-11-18T09:00:00Z"}}}]),
    ]
    gh = fetch_github_repo_summary("openai/math", session=gh_session)
    assert gh is not None
    assert gh.stars == 420
    assert "Add theorem 305" in gh.recent_commits[0]


def test_build_ai_tech_space_context_integrates_space_telemetry():
    session = MagicMock()

    def fake_get(url, **kwargs):
        if "thespacedevs.com" in url:
            return _mock_response(
                200,
                json_data={
                    "results": [
                        {
                            "net": "2026-11-14T08:00:00Z",
                            "name": "Falcon 9 Block 5 | Starlink",
                            "launch_service_provider": {"name": "SpaceX"},
                            "status": {"abbrev": "Success"},
                        }
                    ]
                },
            )
        return _mock_response(200, json_data=[], text="<rss><channel></channel></rss>")

    session.get.side_effect = fake_get
    ctx = build_ai_tech_space_context(
        "How many Falcon 9 launches will SpaceX have in November 2026?",
        session=session,
        now=datetime(2026, 11, 19, tzinfo=timezone.utc),
    )
    assert ctx.playbook == "ai_tech_space"
    assert ctx.subtype == "space_launches"
    assert "LAUNCH LIBRARY 2" in ctx.evidence_text
