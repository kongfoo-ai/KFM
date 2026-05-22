from __future__ import annotations

import logging
import sys
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from web.app.services.opm_validate import (
    humanize_diagram_validation,
    repair_common_llm_link_relations,
    validate_diagram,
)
from web.app.schemas.opm import OpmDiagram

from .conftest import fake_opm_llm_success


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_DIAGRAM: dict = {
    "version": "1.0",
    "nodes": [
        {"id": "farmer", "kind": "object", "label": "Farmer"},
        {"id": "grow", "kind": "process", "label": "Grow"},
        {"id": "crop", "kind": "object", "label": "Crop"},
    ],
    "links": [
        {"id": "l1", "source": "farmer", "target": "grow", "relation": "agent"},
        {"id": "l2", "source": "grow", "target": "crop", "relation": "result"},
    ],
}


def _diagram(**overrides) -> dict:
    """Return a copy of VALID_DIAGRAM with top-level fields overridden."""
    return {**VALID_DIAGRAM, **overrides}


# ---------------------------------------------------------------------------
# Unit tests: valid diagram
# ---------------------------------------------------------------------------


def test_valid_diagram_passes():
    result = validate_diagram(VALID_DIAGRAM)
    assert isinstance(result, OpmDiagram)


def test_valid_diagram_version_preserved():
    result = validate_diagram(VALID_DIAGRAM)
    assert result.version == "1.0"


def test_empty_nodes_list_accepted():
    result = validate_diagram({"version": "1.0", "nodes": [], "links": []})
    assert isinstance(result, OpmDiagram)
    assert result.nodes == []
    assert result.links == []


# ---------------------------------------------------------------------------
# Unit tests: hard rejections
# ---------------------------------------------------------------------------


def test_invalid_node_kind_rejected():
    bad = _diagram(nodes=[{"id": "x", "kind": "thing", "label": "X"}])
    with pytest.raises(ValidationError):
        validate_diagram(bad)


def test_invalid_relation_rejected():
    bad = _diagram(
        links=[{"id": "l1", "source": "farmer", "target": "grow", "relation": "causes"}]
    )
    with pytest.raises(ValidationError):
        validate_diagram(bad)


def test_label_too_long_rejected():
    long_label = "a" * 161
    bad = _diagram(nodes=[{"id": "farmer", "kind": "object", "label": long_label}])
    with pytest.raises(ValidationError):
        validate_diagram(bad)


def test_empty_label_rejected():
    bad = _diagram(nodes=[{"id": "farmer", "kind": "object", "label": ""}])
    with pytest.raises(ValidationError):
        validate_diagram(bad)


def test_invalid_node_id_with_space_rejected():
    bad = _diagram(nodes=[{"id": "farmer node", "kind": "object", "label": "My Node"}])
    with pytest.raises(ValidationError):
        validate_diagram(bad)


def test_humanize_diagram_validation_collapses_many_id_pattern_errors():
    bad = {
        "version": "1.0",
        "nodes": [
            {"id": "ok_a", "kind": "object", "label": "A"},
            {"id": "Bad Upper", "kind": "object", "label": "B"},
            {"id": "also_bad", "kind": "object", "label": "C"},
        ],
        "links": [],
    }
    with pytest.raises(ValidationError) as ei:
        validate_diagram(bad)
    msg = humanize_diagram_validation(ei.value)
    assert "17 validation errors" not in msg
    assert "pydantic" not in msg.lower()
    assert "lowercase" in msg.lower() or "format" in msg.lower()


def test_snake_case_node_id_accepted():
    d = _diagram(
        nodes=[{"id": "my_node_id", "kind": "object", "label": "X"}],
        links=[],
    )
    out = validate_diagram(d)
    assert out.nodes[0].id == "my_node_id"


def test_uppercase_node_id_rejected():
    bad = _diagram(nodes=[{"id": "UPPERCASE", "kind": "object", "label": "X"}])
    with pytest.raises(ValidationError):
        validate_diagram(bad)


def test_duplicate_node_ids_rejected():
    bad = _diagram(
        nodes=[
            {"id": "farmer", "kind": "object", "label": "Farmer"},
            {"id": "farmer", "kind": "process", "label": "Farmer 2"},
        ]
    )
    with pytest.raises((ValidationError, ValueError)):
        validate_diagram(bad)


def test_duplicate_link_ids_renamed_automatically():
    """LLM often reuses the same link id; validation renames duplicates."""
    dup = _diagram(
        links=[
            {"id": "l1", "source": "farmer", "target": "grow", "relation": "agent"},
            {"id": "l1", "source": "grow", "target": "crop", "relation": "result"},
        ]
    )
    result = validate_diagram(dup)
    ids = [lk.id for lk in result.links]
    assert ids == ["l1", "l1-2"]
    assert len(set(ids)) == len(ids)


def test_dangling_link_source_rejected():
    bad = _diagram(
        links=[{"id": "l1", "source": "ghost", "target": "grow", "relation": "agent"}]
    )
    with pytest.raises((ValidationError, ValueError)):
        validate_diagram(bad)


def test_dangling_link_target_rejected():
    bad = _diagram(
        links=[{"id": "l1", "source": "farmer", "target": "ghost", "relation": "agent"}]
    )
    with pytest.raises((ValidationError, ValueError)):
        validate_diagram(bad)


def test_result_object_to_state_rejected():
    """result must be process→object."""
    bad = {
        "version": "1.0",
        "nodes": [
            {"id": "drug", "kind": "object", "label": "Drug"},
            {"id": "adm", "kind": "process", "label": "administer"},
            {"id": "goal", "kind": "state", "label": "management"},
        ],
        "links": [
            {"id": "l1", "source": "drug", "target": "goal", "relation": "result"},
            {"id": "l2", "source": "drug", "target": "adm", "relation": "agent"},
        ],
    }
    with pytest.raises(ValueError, match="result.*process"):
        validate_diagram(bad)


def test_effect_object_to_state_rejected():
    bad = {
        "version": "1.0",
        "nodes": [
            {"id": "drug", "kind": "object", "label": "Drug"},
            {"id": "goal", "kind": "state", "label": "management"},
        ],
        "links": [{"id": "l1", "source": "drug", "target": "goal", "relation": "effect"}],
    }
    with pytest.raises(ValueError, match="effect.*process"):
        validate_diagram(bad)


def test_agent_object_to_state_rejected():
    # "diet" has no human label hint, so agent is repaired to instrument first,
    # then validation rejects instrument→state (must be object→process).
    bad = {
        "version": "1.0",
        "nodes": [
            {"id": "diet", "kind": "object", "label": "diet"},
            {"id": "goal", "kind": "state", "label": "management"},
        ],
        "links": [{"id": "l1", "source": "diet", "target": "goal", "relation": "agent"}],
    }
    with pytest.raises(ValueError, match="object→process"):
        validate_diagram(bad)


def test_repair_swaps_result_to_effect_when_target_is_state():
    """LLM often emits result→state; repair fixes before validate."""
    raw = {
        "version": "1.0",
        "nodes": [
            {"id": "drug", "kind": "object", "label": "Drug"},
            {"id": "adm", "kind": "process", "label": "administer"},
            {"id": "goal", "kind": "state", "label": "outcome"},
        ],
        "links": [
            {"id": "l1", "source": "drug", "target": "adm", "relation": "agent"},
            {"id": "l2", "source": "adm", "target": "goal", "relation": "result"},
        ],
    }
    fixed = repair_common_llm_link_relations(raw)
    assert fixed["links"][1]["relation"] == "effect"
    result = validate_diagram(fixed)
    assert result.links[1].relation.value == "effect"


def test_normalize_numeric_link_endpoint_to_node_prefix():
    raw = {
        "version": "1.0",
        "nodes": [
            {"id": "node5", "kind": "process", "label": "p"},
            {"id": "node6", "kind": "state", "label": "s"},
        ],
        "links": [{"id": "l1", "source": 5, "target": "node6", "relation": "result"}],
    }
    d = validate_diagram(raw)
    assert d.links[0].source == "node5"
    assert d.links[0].relation.value == "effect"


def test_repair_result_to_effect_case_insensitive_relation():
    raw = {
        "version": "1.0",
        "nodes": [
            {"id": "node5", "kind": "process", "label": "p"},
            {"id": "node6", "kind": "state", "label": "s"},
        ],
        "links": [{"id": "link5", "source": "node5", "target": "node6", "relation": "Result"}],
    }
    d = validate_diagram(raw)
    assert d.links[0].relation.value == "effect"


def test_repair_swaps_effect_to_result_when_target_is_object():
    raw = {
        "version": "1.0",
        "nodes": [
            {"id": "p", "kind": "process", "label": "step"},
            {"id": "o", "kind": "object", "label": "product"},
        ],
        "links": [{"id": "l1", "source": "p", "target": "o", "relation": "effect"}],
    }
    fixed = repair_common_llm_link_relations(raw)
    assert fixed["links"][0]["relation"] == "result"
    validate_diagram(fixed)


def test_instrument_object_to_object_rejected():
    bad = {
        "version": "1.0",
        "nodes": [
            {"id": "a", "kind": "object", "label": "A"},
            {"id": "b", "kind": "object", "label": "B"},
        ],
        "links": [{"id": "l1", "source": "a", "target": "b", "relation": "instrument"}],
    }
    with pytest.raises(ValueError, match="instrument.*object"):
        validate_diagram(bad)


def test_humanize_opm_link_rule_errors():
    bad = {
        "version": "1.0",
        "nodes": [
            {"id": "a", "kind": "object", "label": "A"},
            {"id": "b", "kind": "object", "label": "B"},
        ],
        "links": [{"id": "l1", "source": "a", "target": "b", "relation": "instrument"}],
    }
    with pytest.raises(ValidationError) as ei:
        validate_diagram(bad)
    msg = humanize_diagram_validation(ei.value)
    assert "OPM rules" in msg
    assert "object to a process" in msg


# ---------------------------------------------------------------------------
# Unit tests: warning-only semantic checks
# ---------------------------------------------------------------------------


def test_self_loop_warns_not_rejects(caplog):
    diagram = {
        "version": "1.0",
        "nodes": [{"id": "node-a", "kind": "object", "label": "A"}],
        "links": [
            {"id": "l1", "source": "node-a", "target": "node-a", "relation": "aggregation"}
        ],
    }
    with caplog.at_level(logging.WARNING, logger="web.app.services.opm_validate"):
        result = validate_diagram(diagram)
    assert isinstance(result, OpmDiagram)
    assert any("Self-loop" in r.message for r in caplog.records)


def test_duplicate_relation_warns(caplog):
    diagram = {
        "version": "1.0",
        "nodes": [
            {"id": "node-a", "kind": "object", "label": "A"},
            {"id": "node-b", "kind": "process", "label": "B"},
        ],
        "links": [
            {"id": "l1", "source": "node-a", "target": "node-b", "relation": "agent"},
            {"id": "l2", "source": "node-a", "target": "node-b", "relation": "agent"},
        ],
    }
    with caplog.at_level(logging.WARNING, logger="web.app.services.opm_validate"):
        result = validate_diagram(diagram)
    assert isinstance(result, OpmDiagram)
    assert any("Duplicate relation" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Integration tests: router behaviour
# ---------------------------------------------------------------------------


@pytest.fixture()
def _test_engine():
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


@pytest.fixture()
def client(_test_engine) -> Generator:
    sys.modules.setdefault("ollama", MagicMock())
    from fastapi.testclient import TestClient
    from web.app.main import app

    with patch("web.app.db.engine", _test_engine):
        with patch(
            "web.app.services.opm_extract.call_llm", side_effect=fake_opm_llm_success
        ):
            with TestClient(app) as c:
                yield c


def test_valid_stub_passes_validation_and_stores(client):
    response = client.post("/opm/extract", json={"text": "some text", "save_note": False})
    assert response.status_code == 200
    data = response.json()
    assert data["diagram"]["version"] == "1.0"


def test_invalid_diagram_blocked_before_db_insert(client, monkeypatch):
    """An invalid dict from extraction must be rejected with 422, not stored."""
    bad_diagram = {"version": "1.0", "nodes": [{"id": "BadID", "kind": "object", "label": "X"}], "links": []}
    monkeypatch.setattr("web.app.services.opm_extract.extract_opm_diagram", lambda text: bad_diagram)

    response = client.post("/opm/extract", json={"text": "some text", "save_note": False})

    assert response.status_code == 422
    body = response.json()
    assert body["detail"]["stage"] == "validation"
    assert body["detail"]["error"] == "opm_extraction_failed"


def test_invalid_diagram_not_persisted(client, _test_engine, monkeypatch):
    """After a validation failure, no row should appear in opm_diagrams."""
    bad_diagram = {"version": "1.0", "nodes": [{"id": "Bad", "kind": "object", "label": "X"}], "links": []}
    monkeypatch.setattr("web.app.services.opm_extract.extract_opm_diagram", lambda text: bad_diagram)

    client.post("/opm/extract", json={"text": "some text", "save_note": False})

    import web.app.db as db_module
    diagrams = db_module.list_opm_diagrams()
    assert diagrams == []
