"""Tests unitarios offline para el playbook crypto y modelo log-normal."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

from playbooks.crypto import (
    build_crypto_context,
    estimate_lognormal_threshold_prob,
    extract_crypto_tickers,
    extract_usd_target,
)


def _mock_response(status_code: int = 200, *, json_data=None, text: str = ""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    if json_data is not None:
        resp.json.return_value = json_data
    return resp


def test_extract_crypto_tickers_and_usd_targets():
    assert extract_crypto_tickers("Will Bitcoin be at or above $85,000?") == ["BTC-USD"]
    assert extract_crypto_tickers("Will Zcash (ZEC) reach $2,000?") == ["ZEC-USD"]
    assert extract_usd_target("Will Bitcoin be at or above $85,000?") == 85000.0
    assert extract_usd_target("Will Zcash (ZEC) reach $2,000?") == 2000.0


def test_lognormal_threshold_and_barrier_probabilities():
    # Spot igual al target -> probabilidad de cierre arriba ~50%
    p_atm = estimate_lognormal_threshold_prob(85000.0, 85000.0, daily_vol_pct=3.5, days_remaining=7)
    assert p_atm is not None and 0.49 <= p_atm <= 0.51

    # Target muy por encima con poca ventana -> probabilidad baja
    p_otm = estimate_lognormal_threshold_prob(50.0, 2000.0, daily_vol_pct=5.0, days_remaining=2, is_touch_barrier=True)
    assert p_otm is not None and p_otm < 0.02

    # Barrera touch siempre >= cierre para target > spot
    p_close = estimate_lognormal_threshold_prob(80000.0, 90000.0, daily_vol_pct=3.5, days_remaining=7, is_touch_barrier=False)
    p_touch = estimate_lognormal_threshold_prob(80000.0, 90000.0, daily_vol_pct=3.5, days_remaining=7, is_touch_barrier=True)
    assert p_close is not None and p_touch is not None and p_touch > p_close


def test_build_crypto_context_injects_spot_vol_and_lognormal_baseline():
    closes = [80000.0 + i * 250.0 for i in range(25)]
    timestamps = [1760000000 + i * 86400 for i in range(25)]
    session = MagicMock()

    def fake_get(url, **kwargs):
        if "finance.yahoo.com" in url:
            return _mock_response(
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
        return _mock_response(200, json_data=[], text="<rss><channel></channel></rss>")

    session.get.side_effect = fake_get
    ctx = build_crypto_context(
        "Will Bitcoin be at or above $85,000 on November 25, 2026?",
        session=session,
        now=datetime(2026, 11, 19, tzinfo=timezone.utc),
    )
    assert ctx.playbook == "crypto"
    assert ctx.subtype == "price_volatility"
    assert "BTC-USD" in ctx.evidence_text
    assert "Log-normal" in ctx.evidence_text
    assert ctx.market_prior is not None
