"""Conectores climáticos y ambientales oficiales (0 tokens LLM, sin API key).

Fuentes verificadas en vivo (200 OK):
- NOAA CPC ONI (oni.ascii.txt): índice trimestral oficial de El Niño / La Niña.
- US Drought Monitor API (usdmdataservices.unl.edu): % semanal oficial de sequía en EE.UU.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import requests

from playbooks.open_sources import DEFAULT_HEADERS, HTTP_TIMEOUT_SECONDS

NOAA_ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
US_DROUGHT_API_URL = (
    "https://usdmdataservices.unl.edu/api/USStatistics/"
    "GetDroughtSeverityStatisticsByAreaPercent"
)


@dataclass(frozen=True)
class OniReading:
    """Lectura trimestral oficial del Oceanic Niño Index (NOAA CPC)."""

    season: str
    year: int
    total_sst: float
    anomaly: float


@dataclass(frozen=True)
class DroughtReading:
    """Lectura semanal oficial del US Drought Monitor (CONUS)."""

    map_date: str
    none_pct: float
    d0_or_worse_pct: float
    d1_drought_pct: float
    d2_severe_pct: float


def fetch_noaa_oni_latest(
    *,
    session: requests.Session | None = None,
    tail: int = 4,
) -> list[OniReading]:
    """Descarga y parsea las últimas lecturas trimestrales oficiales del ONI."""
    client = session or requests.Session()
    try:
        resp = client.get(NOAA_ONI_URL, headers=DEFAULT_HEADERS, timeout=HTTP_TIMEOUT_SECONDS)
        if resp.status_code != 200:
            return []
    except requests.RequestException:
        return []

    readings: list[OniReading] = []
    for line in resp.text.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            readings.append(
                OniReading(
                    season=parts[0],
                    year=int(parts[1]),
                    total_sst=float(parts[2]),
                    anomaly=float(parts[3]),
                )
            )
        except ValueError:
            continue
    return readings[-tail:]


def fetch_us_drought_latest(
    *,
    session: requests.Session | None = None,
    now: datetime | None = None,
    tail: int = 3,
) -> list[DroughtReading]:
    """Descarga las últimas lecturas semanales de sequía CONUS desde US Drought Monitor."""
    ref = now or datetime.now(timezone.utc)
    start_str = (ref - timedelta(days=35)).strftime("%m/%d/%Y")
    end_str = ref.strftime("%m/%d/%Y")

    client = session or requests.Session()
    try:
        resp = client.get(
            US_DROUGHT_API_URL,
            params={
                "aoi": "conus",
                "startdate": start_str,
                "enddate": end_str,
                "statisticsType": 1,
            },
            headers=DEFAULT_HEADERS,
            timeout=HTTP_TIMEOUT_SECONDS,
        )
        if resp.status_code != 200:
            return []
    except requests.RequestException:
        return []

    lines = [ln.strip() for ln in resp.text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return []

    readings: list[DroughtReading] = []
    for row in lines[1:]:
        cols = row.split(",")
        if len(cols) < 8:
            continue
        try:
            readings.append(
                DroughtReading(
                    map_date=cols[0],
                    none_pct=float(cols[2]),
                    d0_or_worse_pct=float(cols[3]),
                    d1_drought_pct=float(cols[4]),
                    d2_severe_pct=float(cols[5]),
                )
            )
        except ValueError:
            continue

    # Orden cronológico ascendente (el endpoint suele devolver más reciente primero)
    readings.sort(key=lambda r: r.map_date)
    return readings[-tail:]
