"""Tests unitarios offline para el playbook politics_geo_law y fuentes abiertas."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

from playbooks.open_sources import (
    fetch_bluesky_watchlist,
    fetch_google_news_rss,
    fetch_manifold_prior,
)
from playbooks.politics_geo_law import (
    build_politics_geo_law_context,
    estimate_days_remaining,
)
from playbooks.watchlists import detect_politics_subtype


def _mock_response(status_code: int = 200, *, json_data=None, text: str = ""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    if json_data is not None:
        resp.json.return_value = json_data
    return resp


def test_detect_politics_subtype_maps_all_four_patterns():
    assert (
        detect_politics_subtype(
            "Which candidate will lead Alaska's U.S. Senate race?",
            ["elections"],
        )
        == "elections"
    )
    assert (
        detect_politics_subtype(
            "How many drones will Ukraine's Air Force report Russia launched?",
            ["geopolitics"],
        )
        == "conflict_geo"
    )
    assert (
        detect_politics_subtype(
            "Will James Comey have at least two federal criminal cases pending?",
            ["law"],
        )
        == "law_courts"
    )
    assert (
        detect_politics_subtype(
            "Will SRS or DOE have publicly stated a cause for the Savannah River fire?",
            ["nuclear technology & risks"],
        )
        == "parliament_bureaucracy"
    )


def test_estimate_days_remaining_from_question_text():
    now = datetime(2026, 11, 1, tzinfo=timezone.utc)
    days = estimate_days_remaining(
        "Will DOE state a cause by November 13, 2026?",
        now=now,
    )
    assert days == 12


def test_fetch_manifold_prior_filters_by_traders_and_overlap():
    session = MagicMock()
    session.get.return_value = _mock_response(
        200,
        json_data=[
            {
                "question": "Will Alaska Senate race be led by incumbent?",
                "isResolved": False,
                "uniqueBettorCount": 42,
                "probability": 0.62,
                "volume": 3500,
            },
            {
                "question": "Will Alaska Senate race be led by incumbent?",
                "isResolved": False,
                "uniqueBettorCount": 2,
                "probability": 0.90,
                "volume": 10,
            },
        ],
    )
    matches = fetch_manifold_prior(
        "Which candidate will lead Alaska Senate race?",
        session=session,
    )
    assert len(matches) == 1
    assert matches[0].probability == 0.62
    assert matches[0].source == "Manifold"


def test_fetch_google_news_rss_parses_items_and_deduplicates():
    rss_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0"><channel>
      <item>
        <title>Alaska Senate race narrows in early count</title>
        <pubDate>Thu, 12 Nov 2026 14:00:00 GMT</pubDate>
        <source>AP News</source>
      </item>
      <item>
        <title>Alaska Senate race narrows in early count</title>
        <pubDate>Thu, 12 Nov 2026 14:05:00 GMT</pubDate>
        <source>AP News</source>
      </item>
      <item>
        <title>Election officials update Alaska tally</title>
        <pubDate>Thu, 12 Nov 2026 15:00:00 GMT</pubDate>
        <source>Reuters</source>
      </item>
      <item>
        <title>Senate control hinges on western states</title>
        <pubDate>Thu, 12 Nov 2026 16:00:00 GMT</pubDate>
        <source>Politico</source>
      </item>
    </channel></rss>"""
    session = MagicMock()
    session.get.return_value = _mock_response(200, text=rss_xml)

    items = fetch_google_news_rss(
        "Alaska Senate race count",
        domains=("apnews.com", "reuters.com"),
        session=session,
    )
    assert len(items) == 3
    assert items[0].published == "2026-11-12"
    assert items[0].source == "AP News"


def test_fetch_bluesky_watchlist_filters_relevant_posts():
    session = MagicMock()
    session.get.return_value = _mock_response(
        200,
        json_data={
            "feed": [
                {
                    "post": {
                        "record": {
                            "text": "Ukraine Air Force reports 95 Russia drones launched overnight",
                            "createdAt": "2026-11-13T07:30:00Z",
                        }
                    }
                },
                {
                    "post": {
                        "record": {
                            "text": "Unrelated local sports update",
                            "createdAt": "2026-11-13T07:00:00Z",
                        }
                    }
                },
            ]
        },
    )
    items = fetch_bluesky_watchlist(
        "How many drones will Ukraine Air Force report Russia launched?",
        handles=("kyivindependent.com",),
        session=session,
    )
    assert len(items) == 1
    assert items[0].source == "@kyivindependent.com"
    assert "95 Russia drones" in items[0].text


def test_build_politics_geo_law_context_assembles_tiers_and_caps_length():
    session = MagicMock()

    def fake_get(url, **kwargs):
        if "polymarket" in url:
            return _mock_response(200, json_data={"events": []})
        if "manifold" in url:
            return _mock_response(
                200,
                json_data=[
                    {
                        "question": "Will DOE publicly state cause for Savannah River fire?",
                        "isResolved": False,
                        "uniqueBettorCount": 25,
                        "probability": 0.22,
                        "volume": 900,
                    }
                ],
            )
        if "news.google.com" in url:
            return _mock_response(
                200,
                text="<rss><channel><item><title>DOE reviews Savannah River fire</title>"
                "<pubDate>Wed, 11 Nov 2026 10:00:00 GMT</pubDate>"
                "<source>Reuters</source></item></channel></rss>",
            )
        return _mock_response(200, json_data={"feed": []})

    session.get.side_effect = fake_get
    ctx = build_politics_geo_law_context(
        "Will DOE publicly state a cause for the Savannah River fire by November 13, 2026?",
        official_categories=["nuclear technology & risks"],
        session=session,
        now=datetime(2026, 11, 10, tzinfo=timezone.utc),
    )
    assert ctx.subtype == "parliament_bureaucracy"
    assert ctx.days_remaining == 3
    assert ctx.market_prior == 0.22
    assert ctx.market_prior_source == "Manifold"
    assert ctx.apply_status_quo_bias is True
    assert "TIER 1" in ctx.evidence_text
    assert "TIER 2" in ctx.evidence_text
    assert "Pre-mortem" in ctx.prompt_addendum


def test_dispatcher_widens_numeric_and_date_percentiles_and_calibrates():
    from dataclasses import dataclass
    from playbooks.dispatcher import (
        calibrate_binary_with_context,
        widen_percentile_objects,
    )

    @dataclass
    class FakePercentile:
        percentile: float
        value: float

    @dataclass
    class FakeDateQuestion:
        lower_bound: datetime
        upper_bound: datetime
        open_lower_bound: bool = False
        open_upper_bound: bool = False

    q = FakeDateQuestion(
        lower_bound=datetime(2026, 1, 1, tzinfo=timezone.utc),
        upper_bound=datetime(2028, 1, 1, tzinfo=timezone.utc),
    )
    raw = [
        FakePercentile(0.1, datetime(2026, 6, 1, tzinfo=timezone.utc).timestamp()),
        FakePercentile(0.4, datetime(2026, 9, 1, tzinfo=timezone.utc).timestamp()),
        FakePercentile(0.6, datetime(2026, 11, 1, tzinfo=timezone.utc).timestamp()),
        FakePercentile(0.9, datetime(2027, 2, 1, tzinfo=timezone.utc).timestamp()),
    ]
    widened = widen_percentile_objects(raw, q, inflation_factor=1.25)
    assert widened[0].value < raw[0].value
    assert widened[-1].value > raw[-1].value

    cal = calibrate_binary_with_context(0.99, None)
    assert cal.calibrated_probability <= 0.98
