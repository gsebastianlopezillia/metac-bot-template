"""Tests unitarios offline para health_science y conectores científicos."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

from playbooks.health_science import (
    build_health_science_context,
    detect_health_science_subtype,
)
from playbooks.science_sources import (
    detect_regional_locale,
    fetch_arxiv_preprints,
    fetch_clinical_trials_summary,
    fetch_regional_news_rss,
)


def _mock_response(status_code: int = 200, *, json_data=None, text: str = ""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    if json_data is not None:
        resp.json.return_value = json_data
    return resp


def test_detect_health_science_subtype_and_regional_locale():
    assert detect_health_science_subtype("How many new suspected measles cases in Bangladesh?") == "epidemiology_outbreaks"
    assert detect_health_science_subtype("Will FDA approve the Phase 3 vaccine trial?") == "clinical_regulatory"
    assert detect_health_science_subtype("Will authors post a manuscript proving Yang-Mills?") == "science_preprints"
    assert detect_health_science_subtype("Will Hospital Privado de Mendoza operating rooms be open?") == "local_health_ops"

    assert detect_regional_locale("Hospital Privado de Mendoza surgeries") == ("es-419", "AR", "AR:es-419")
    assert detect_regional_locale("InfoGripe bulletin Brazil SRAG") == ("pt-BR", "BR", "BR:pt-419")
    assert detect_regional_locale("US CDC flu surveillance") is None


def test_fetch_arxiv_preprints_parses_atom_feed():
    atom_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <title>Notes on four-dimensional Yang-Mills fields</title>
        <published>2026-11-10T18:00:00Z</published>
        <author><name>Levent Alpoge</name></author>
      </entry>
    </feed>"""
    session = MagicMock()
    session.get.return_value = _mock_response(200, text=atom_xml)

    papers = fetch_arxiv_preprints("Levent Alpoge Yang-Mills manuscript", session=session)
    assert len(papers) == 1
    assert papers[0].published == "2026-11-10"
    assert "Yang-Mills" in papers[0].title


def test_fetch_clinical_trials_summary_extracts_count_and_titles():
    session = MagicMock()
    session.get.return_value = _mock_response(
        200,
        json_data={
            "totalCount": 14,
            "studies": [
                {"protocolSection": {"identificationModule": {"briefTitle": "LLM Chatbot Trial A"}}},
                {"protocolSection": {"identificationModule": {"briefTitle": "LLM Chatbot Trial B"}}},
            ],
        },
    )
    summary = fetch_clinical_trials_summary("LLM Chatbot clinical trial recruiting", session=session, recruiting_only=True)
    assert summary is not None
    assert summary.total_count == 14
    assert len(summary.sample_titles) == 2


def test_fetch_regional_news_rss_uses_local_edition():
    rss_xml = """<rss><channel><item>
      <title>Hospital Privado de Mendoza reabre quirófanos tras inspección</title>
      <pubDate>Thu, 12 Nov 2026 10:00:00 GMT</pubDate>
      <source>Los Andes</source>
    </item></channel></rss>"""
    session = MagicMock()
    session.get.return_value = _mock_response(200, text=rss_xml)

    items = fetch_regional_news_rss(
        "Will the operating rooms of Hospital Privado de Mendoza be open?",
        session=session,
    )
    assert len(items) == 1
    assert items[0].tier == "TIER 2 - REGIONAL PRESS (AR)"
    assert items[0].source == "Los Andes"


def test_build_health_science_context_integrates_arxiv_and_news():
    session = MagicMock()

    def fake_get(url, **kwargs):
        if "arxiv.org" in url:
            return _mock_response(
                200,
                text='<feed xmlns="http://www.w3.org/2005/Atom">'
                "<entry><title>Preliminary Yang-Mills notes</title>"
                "<published>2026-11-05T00:00:00Z</published>"
                "<author><name>Ethan Sussman</name></author></entry></feed>",
            )
        return _mock_response(200, json_data=[], text="<rss><channel></channel></rss>")

    session.get.side_effect = fake_get
    ctx = build_health_science_context(
        "Will Ethan Sussman post a manuscript proving Yang-Mills by November 13, 2026?",
        session=session,
        now=datetime(2026, 11, 10, tzinfo=timezone.utc),
    )
    assert ctx.playbook == "health_science"
    assert ctx.subtype == "science_preprints"
    assert "TIER 1 - ARXIV PREPRINTS" in ctx.evidence_text
    assert ctx.apply_status_quo_bias is True
