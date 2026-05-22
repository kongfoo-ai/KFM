from __future__ import annotations

import json
import sys
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from ..app.services.opm_extract import (
    OPMExtractionError,
    extract_opm_diagram,
    normalize_llm_output,
    parse_json,
)
from .conftest import VALID_OPM_LLM_JSON, fake_opm_llm_success


# ---------------------------------------------------------------------------
# Unit tests: JSON helpers
# ---------------------------------------------------------------------------


def test_parse_json_accepts_object():
    assert parse_json('{"a": 1}') == {"a": 1}


def test_parse_json_invalid_raises():
    with pytest.raises(ValueError, match="Invalid JSON"):
        parse_json("not json")


def test_parse_json_array_raises_must_be_object():
    with pytest.raises(ValueError, match="JSON must be object"):
        parse_json("[1,2]")


def test_normalize_strips_markdown_fence():
    raw = '```json\n{"version": "1.0", "nodes": [], "links": []}\n```'
    out = normalize_llm_output(raw)
    assert parse_json(out)["version"] == "1.0"


# ---------------------------------------------------------------------------
# Unit tests: system prompt content
# ---------------------------------------------------------------------------


def test_system_prompt_contains_pharma_object_vocab():
    from web.app.services.opm_extract import build_system_prompt
    prompt = build_system_prompt()
    for term in ("receptor", "metabolite", "inhibitor"):
        assert term in prompt, f"Expected pharma object term '{term}' in system prompt"


def test_system_prompt_contains_pharma_process_vocab():
    from web.app.services.opm_extract import build_system_prompt
    prompt = build_system_prompt()
    for term in ("metabolising", "administering", "activating"):
        assert term in prompt, f"Expected pharma process term '{term}' in system prompt"


def test_system_prompt_contains_pharma_state_vocab():
    from web.app.services.opm_extract import build_system_prompt
    prompt = build_system_prompt()
    for term in ("adverse event", "toxicity", "bioavailability"):
        assert term in prompt, f"Expected pharma state term '{term}' in system prompt"


def test_system_prompt_contains_moa_example():
    from web.app.services.opm_extract import build_system_prompt
    prompt = build_system_prompt()
    assert "glp1_agonist" in prompt, "Expected mechanism-of-action few-shot example in system prompt"


def test_system_prompt_contains_adverse_event_example():
    from web.app.services.opm_extract import build_system_prompt
    prompt = build_system_prompt()
    assert "warfarin" in prompt, "Expected adverse-event few-shot example in system prompt"


# ---------------------------------------------------------------------------
# Unit tests: default model
# ---------------------------------------------------------------------------


def test_call_llm_defaults_to_gpt4o_mini_when_model_unset(monkeypatch):
    """When OPM_MODEL is not set, call_llm should use gpt-4o-mini."""
    monkeypatch.delenv("OPM_MODEL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    mock_message = MagicMock()
    mock_message.content = '{"version":"1.0","nodes":[],"links":[]}'
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    with patch("web.app.services.opm_extract.OpenAI") as MockOpenAI:
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_response

        from web.app.services.opm_extract import call_llm
        call_llm("sys", "user")

        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "gpt-4o-mini", (
            f"Expected model 'gpt-4o-mini', got '{call_kwargs.kwargs['model']}'"
        )


# ---------------------------------------------------------------------------
# Unit tests: extract (mocked LLM)
# ---------------------------------------------------------------------------


def test_extract_returns_dict_with_mock_llm():
    with patch("web.app.services.opm_extract.call_llm", side_effect=fake_opm_llm_success):
        result = extract_opm_diagram("any text")
    assert isinstance(result, dict)


def test_extract_version_field():
    with patch("web.app.services.opm_extract.call_llm", side_effect=fake_opm_llm_success):
        result = extract_opm_diagram("any text")
    assert result["version"] == "1.0"


def test_extract_has_nodes_and_links():
    with patch("web.app.services.opm_extract.call_llm", side_effect=fake_opm_llm_success):
        result = extract_opm_diagram("any text")
    assert "nodes" in result
    assert "links" in result


def test_extract_non_object_json_no_retry_raises_422():
    calls: list[None] = []

    def bad_llm(s: str, u: str, **kwargs) -> str:
        calls.append(None)
        return "[1]"

    with patch("web.app.services.opm_extract.call_llm", side_effect=bad_llm):
        with pytest.raises(OPMExtractionError) as ei:
            extract_opm_diagram("x")
    assert ei.value.status_code == 422
    assert "JSON must be object" in ei.value.detail
    assert len(calls) == 1


def test_extract_invalid_json_retries_once_then_422():
    calls: list[int] = []

    def flaky(s: str, u: str, **kwargs) -> str:
        calls.append(1)
        if len(calls) == 1:
            return "not-json"
        return VALID_OPM_LLM_JSON

    with patch("web.app.services.opm_extract.call_llm", side_effect=flaky):
        result = extract_opm_diagram("x")
    assert result["version"] == "1.0"
    assert len(calls) == 2


def test_extract_502_retries_once():
    attempts = {"n": 0}

    def fail_then_ok(s: str, u: str, **kwargs) -> str:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise OPMExtractionError(502, "timeout", None)
        return VALID_OPM_LLM_JSON

    with patch("web.app.services.opm_extract.call_llm", side_effect=fail_then_ok):
        result = extract_opm_diagram("x")
    assert result["version"] == "1.0"
    assert attempts["n"] == 2


def test_extract_validation_failure_triggers_second_llm_round():
    """Graph integrity (e.g. duplicate node ids) fails → repair prompt → valid JSON."""
    calls: list[str] = []
    # Loose validation accepts object→object instrument; use duplicate node ids (still rejected).
    bad = json.dumps(
        {
            "version": "1.0",
            "nodes": [
                {"id": "a", "kind": "object", "label": "A"},
                {"id": "a", "kind": "object", "label": "B"},
            ],
            "links": [],
        }
    )

    def bad_then_ok(s: str, u: str, **kwargs) -> str:
        calls.append(u)
        if len(calls) == 1:
            return bad
        return VALID_OPM_LLM_JSON

    with patch("web.app.services.opm_extract.call_llm", side_effect=bad_then_ok):
        result = extract_opm_diagram("x")
    assert result["version"] == "1.0"
    assert len(calls) == 2
    assert "failed server validation" in calls[1]


# ---------------------------------------------------------------------------
# Unit tests: JSON round-trip
# ---------------------------------------------------------------------------


def test_payload_round_trip():
    original = {
        "version": "1.0",
        "nodes": [{"id": "example-object", "kind": "object", "label": "example"}],
        "links": [],
    }
    encoded = json.dumps(original)
    decoded = json.loads(encoded)
    assert decoded == original


def test_version_preserved_in_round_trip():
    payload = {"version": "1.0", "nodes": [], "links": []}
    assert json.loads(json.dumps(payload))["version"] == "1.0"


# ---------------------------------------------------------------------------
# Fixtures: in-memory DB for isolation
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_db():
    from sqlalchemy import create_engine as _create_engine
    from sqlalchemy.pool import StaticPool
    test_engine = _create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with patch("web.app.db.engine", test_engine):
        import web.app.db as db_module
        db_module.init_db()
        yield test_engine


# ---------------------------------------------------------------------------
# Unit tests: DB helpers
# ---------------------------------------------------------------------------


def test_insert_opm_diagram_returns_int(tmp_db):
    from web.app import db as db_module
    payload = {"version": "1.0", "nodes": [], "links": []}
    diagram_id = db_module.insert_opm_diagram(payload)
    assert isinstance(diagram_id, int)


def test_insert_opm_diagram_multiple_distinct_ids(tmp_db):
    from web.app import db as db_module
    payload = {"version": "1.0", "nodes": [], "links": []}
    id1 = db_module.insert_opm_diagram(payload)
    id2 = db_module.insert_opm_diagram(payload)
    assert id1 != id2


def test_get_opm_diagram_returns_parsed_dict(tmp_db):
    from web.app import db as db_module
    payload = {"version": "1.0", "nodes": [{"id": "x"}], "links": []}
    diagram_id = db_module.insert_opm_diagram(payload)
    row = db_module.get_opm_diagram(diagram_id)
    assert row is not None
    assert isinstance(row["diagram"], dict)
    assert row["diagram"]["version"] == "1.0"


def test_get_opm_diagram_not_found_returns_none(tmp_db):
    from web.app import db as db_module
    assert db_module.get_opm_diagram(99999) is None


def test_note_id_null_when_not_provided(tmp_db):
    from web.app import db as db_module
    payload = {"version": "1.0", "nodes": [], "links": []}
    diagram_id = db_module.insert_opm_diagram(payload)
    row = db_module.get_opm_diagram(diagram_id)
    assert row is not None
    assert row["note_id"] is None


def test_list_opm_diagrams_returns_all(tmp_db):
    from web.app import db as db_module
    payload = {"version": "1.0", "nodes": [], "links": []}
    db_module.insert_opm_diagram(payload)
    db_module.insert_opm_diagram(payload)
    diagrams = db_module.list_opm_diagrams()
    assert len(diagrams) == 2


def test_stored_payload_matches_inserted(tmp_db):
    from web.app import db as db_module
    payload = {"version": "1.0", "nodes": [{"id": "n1"}], "links": []}
    diagram_id = db_module.insert_opm_diagram(payload)
    row = db_module.get_opm_diagram(diagram_id)
    assert row is not None
    assert row["diagram"] == payload


# ---------------------------------------------------------------------------
# Integration tests: API
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(tmp_db) -> Generator[TestClient, None, None]:
    sys.modules.setdefault("ollama", MagicMock())
    from web.app.main import app

    with patch("web.app.db.engine", tmp_db):
        with patch(
            "web.app.services.opm_extract.call_llm", side_effect=fake_opm_llm_success
        ):
            with TestClient(app) as c:
                yield c


def test_post_extract_inserts_row(client: TestClient, tmp_db):
    from web.app import db as db_module
    response = client.post("/opm/extract", json={"text": "some text", "save_note": False})
    assert response.status_code == 200
    data = response.json()
    assert "diagram_id" in data
    row = db_module.get_opm_diagram(data["diagram_id"])
    assert row is not None


def test_post_extract_response_shape(client: TestClient):
    response = client.post("/opm/extract", json={"text": "some text", "save_note": False})
    assert response.status_code == 200
    data = response.json()
    assert "note_id" in data
    assert "diagram_id" in data
    assert "diagram" in data
    diagram = data["diagram"]
    assert "version" in diagram
    assert "nodes" in diagram
    assert "links" in diagram


def test_post_extract_note_id_null_when_save_note_false(client: TestClient):
    response = client.post("/opm/extract", json={"text": "some text", "save_note": False})
    assert response.status_code == 200
    assert response.json()["note_id"] is None


def test_get_by_id_returns_same_payload(client: TestClient):
    post_resp = client.post("/opm/extract", json={"text": "some text", "save_note": False})
    diagram_id = post_resp.json()["diagram_id"]
    inserted_diagram = post_resp.json()["diagram"]

    get_resp = client.get(f"/opm/{diagram_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["diagram"] == inserted_diagram


def test_get_by_id_404_when_missing(client: TestClient):
    response = client.get("/opm/99999")
    assert response.status_code == 404


def test_get_list_returns_diagrams_key(client: TestClient):
    client.post("/opm/extract", json={"text": "a", "save_note": False})
    response = client.get("/opm")
    assert response.status_code == 200
    assert "diagrams" in response.json()


def test_multiple_inserts_distinct_ids(client: TestClient):
    r1 = client.post("/opm/extract", json={"text": "a", "save_note": False})
    r2 = client.post("/opm/extract", json={"text": "b", "save_note": False})
    assert r1.json()["diagram_id"] != r2.json()["diagram_id"]


def test_post_extract_missing_text_returns_400(client: TestClient):
    response = client.post("/opm/extract", json={"save_note": False})
    assert response.status_code == 400


def test_post_extract_llm_error_returns_opm_extraction_failed(client: TestClient):
    # client fixture reloads web.app; use OPMExtractionError from that same module instance
    import web.app.services.opm_extract as opm_mod

    err = opm_mod.OPMExtractionError(502, "LLM unavailable", None)
    with patch.object(opm_mod, "call_llm", side_effect=err):
        response = client.post("/opm/extract", json={"text": "hello", "save_note": False})
    assert response.status_code == 502
    detail = response.json()["detail"]
    assert detail["error"] == "opm_extraction_failed"
    assert detail["stage"] == "llm"
    assert detail["detail"] == "LLM unavailable"
