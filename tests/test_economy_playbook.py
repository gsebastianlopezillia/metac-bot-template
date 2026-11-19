"""Tests unitarios offline para economy_business y conectores financieros."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

from playbooks.economy_business import (
    build_economy_business_context,
    detect_economy_subtype,
)
from playbooks.finance_sources import (
    extract_fred_series_ids,
    extract_tickers,
    fetch_alpaca_news,
    fetch_fred_series,
    fetch_yahoo_stats,
)


def _mock_response(status_code: int = 200, *, json_data=None, text: str = ""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    if json_data is not None:
        resp.json.return_value = json_data
    return resp


def test_extract_tickers_and_fred_series():
    q = "By how many percentage points will Okta (OKTA) outperform the software ETF IGV?"
    assert extract_tickers(q) == ["OKTA", "IGV"]

    q_company = "Will Applied Materials' Q1 FY2027 revenue guidance midpoint be above $11B?"
    assert extract_tickers(q_company) == ["AMAT"]

    fred_q = "Resolves strictly using FRED series FYONET (Federal Net Outlays)."
    assert extract_fred_series_ids(fred_q) == ["FYONET"]


def test_detect_economy_subtype_maps_all_four_patterns():
    assert detect_economy_subtype("Will Berkshire Hathaway increase Alphabet in Q3 13F?") == "earnings_filings"
    assert detect_economy_subtype("By how much will Okta (OKTA) outperform ETF IGV?") == "equities_markets"
    assert detect_economy_subtype("What will IEA Oil Market Report say on oil demand?") == "macro_fred_commodities"
    assert detect_economy_subtype("Will AST publish the practical-driving-test calendar?") == "corporate_events"


def test_fetch_yahoo_stats_computes_volatility_and_returns():
    closes = [100.0 + i * 0.5 for i in range(25)]
    timestamps = [1760000000 + i * 86400 for i in range(25)]
    session = MagicMock()
    session.get.return_value = _mock_response(
        200,
        json_data={
            "chart": {
                "result": [
                    {
                        "timestamp": timestamps,
                        "indicators": {"quote": [{"close": closes}]},
                    }
                ]
            }
        },
    )
    stats = fetch_yahoo_stats("OKTA", session=session)
    assert stats is not None
    assert stats.ticker == "OKTA"
    assert stats.last_close == 112.0
    assert stats.ret_14d_pct is not None and stats.ret_14d_pct > 0
    assert stats.daily_vol_pct is not None and stats.daily_vol_pct > 0


def test_fetch_fred_series_parses_csv_and_ignores_missing_dots():
    csv_text = "observation_date,FYONET\n2022-01-01,6273324\n2023-01-01,.\n2024-01-01,6750000\n2025-01-01,7020000\n"
    session = MagicMock()
    session.get.return_value = _mock_response(200, text=csv_text)

    summary = fetch_fred_series("FYONET", session=session)
    assert summary is not None
    assert summary.series_id == "FYONET"
    assert summary.observations[-1] == ("2025-01-01", 7020000.0)
    assert summary.yoy_change_pct == 4.0


def test_fetch_alpaca_news_skips_without_keys_and_parses_with_keys(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    assert fetch_alpaca_news(["AMZN"]) == []

    monkeypatch.setenv("ALPACA_API_KEY", "test_key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test_secret")
    session = MagicMock()
    session.get.return_value = _mock_response(
        200,
        json_data={
            "news": [
                {
                    "headline": "Amazon posts strong AWS growth in Q3",
                    "created_at": "2026-10-29T21:05:00Z",
                    "source": "benzinga",
                }
            ]
        },
    )
    items = fetch_alpaca_news(["AMZN"], session=session)
    assert len(items) == 1
    assert items[0].source == "benzinga"
    assert "AWS growth" in items[0].text


def test_build_economy_business_context_integrates_fred_and_news():
    session = MagicMock()

    def fake_get(url, **kwargs):
        if "fredgraph.csv" in url:
            return _mock_response(200, text="date,FYONET\n2024-01-01,6750000\n2025-01-01,7100000\n")
        if "news.google.com" in url:
            return _mock_response(
                200,
                text="<rss><channel><item><title>US federal outlays rise in fiscal 2026</title>"
                "<pubDate>Tue, 10 Nov 2026 12:00:00 GMT</pubDate>"
                "<source>Reuters</source></item></channel></rss>",
            )
        return _mock_response(200, json_data=[])

    session.get.side_effect = fake_get
    ctx = build_economy_business_context(
        "What will US federal net outlays be in FY2028?",
        resolution_criteria="Resolves via FRED series FYONET.",
        session=session,
        now=datetime(2026, 11, 19, tzinfo=timezone.utc),
    )
    assert ctx.playbook == "economy_business"
    assert ctx.subtype == "macro_fred_commodities"
    assert "TIER 1 - OFFICIAL MACRO DATA (FRED)" in ctx.evidence_text
    assert "7.1e+06" in ctx.evidence_text or "7100000" in ctx.evidence_text
