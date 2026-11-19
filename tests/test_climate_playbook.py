"""Tests unitarios offline para climate_environment y conectores climáticos."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

from playbooks.climate_environment import (
    build_climate_environment_context,
    detect_climate_subtype,
)
from playbooks.climate_sources import (
    fetch_noaa_oni_latest,
    fetch_us_drought_latest,
)


def _mock_response(status_code: int = 200, *, json_data=None, text: str = ""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    if json_data is not None:
        resp.json.return_value = json_data
    return resp


def test_detect_climate_subtype_maps_all_four_patterns():
    assert detect_climate_subtype("Will the August-October 2026 El Niño index be 2.10°C?") == "enso_weather_extremes"
    assert detect_climate_subtype("Will under 58% of the contiguous U.S. be in drought?") == "drought_hydrology"
    assert detect_climate_subtype("What will the EU aggregate gas storage filling level be?") == "energy_emissions"
    assert detect_climate_subtype("Will CBD COP17 in Yerevan be suspended?") == "climate_diplomacy_ecology"


def test_fetch_noaa_oni_latest_parses_official_ascii_table():
    ascii_text = (
        " SEAS  YR   TOTAL   ANOM\n"
        "  JJA 2026  29.10   1.85\n"
        "  JAS 2026  29.40   2.15\n"
        "  ASO 2026  29.60   2.35\n"
    )
    session = MagicMock()
    session.get.return_value = _mock_response(200, text=ascii_text)

    readings = fetch_noaa_oni_latest(session=session, tail=2)
    assert len(readings) == 2
    assert readings[-1].season == "ASO"
    assert readings[-1].anomaly == 2.35


def test_fetch_us_drought_latest_parses_conus_csv_in_chronological_order():
    csv_text = (
        "MapDate,AreaOfInterest,None,D0,D1,D2,D3,D4,ValidStart,ValidEnd,StatisticFormatID\r\n"
        "20261117,CONUS,35.15,64.85,42.03,20.84,8.77,2.42,2026-11-17,2026-11-23,1\r\n"
        "20261110,CONUS,34.00,66.00,43.50,21.00,9.00,2.50,2026-11-10,2026-11-16,1\r\n"
    )
    session = MagicMock()
    session.get.return_value = _mock_response(200, text=csv_text)

    readings = fetch_us_drought_latest(session=session, now=datetime(2026, 11, 19, tzinfo=timezone.utc))
    assert len(readings) == 2
    assert readings[0].map_date == "20261110"
    assert readings[-1].d1_drought_pct == 42.03


def test_build_climate_context_injects_drought_data():
    session = MagicMock()

    def fake_get(url, **kwargs):
        if "usdmdataservices" in url:
            return _mock_response(
                200,
                text="MapDate,AreaOfInterest,None,D0,D1,D2,D3,D4\n20261117,CONUS,35.15,64.85,42.03,20.84,8.77,2.42\n",
            )
        return _mock_response(200, json_data=[], text="<rss><channel></channel></rss>")

    session.get.side_effect = fake_get
    ctx = build_climate_environment_context(
        "Will under 58% of the contiguous U.S. be in drought on October 27, 2026?",
        session=session,
        now=datetime(2026, 11, 19, tzinfo=timezone.utc),
    )
    assert ctx.playbook == "climate_environment"
    assert ctx.subtype == "drought_hydrology"
    assert "US DROUGHT MONITOR" in ctx.evidence_text
    assert "42.03%" in ctx.evidence_text
