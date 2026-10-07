import json

import pytest

from building_with_rag.contracts import RetrievedChunk
from building_with_rag.generation.answer import (
    MalformedOutput,
    _resolve,
    parse_model_output,
)

LABELS = {"E1", "E2"}


def _payload(**over):
    data = {
        "outcome": "answered",
        "answer": "Theft is punishable.",
        "claims": [{"text": "Theft punished", "evidence": ["E2"]}],
        "reason": "",
    }
    data.update(over)
    return json.dumps(data)


def test_unknown_label_malformed():
    bad = _payload(claims=[{"text": "x", "evidence": ["E9"]}])
    with pytest.raises(MalformedOutput):
        parse_model_output(bad, LABELS)


def test_non_json_malformed():
    with pytest.raises(MalformedOutput):
        parse_model_output("not json", LABELS)


def test_answered_without_claims_malformed():
    with pytest.raises(MalformedOutput):
        parse_model_output(_payload(claims=[]), LABELS)


def test_valid_answered_resolves_citations():
    parsed = parse_model_output("```json\n" + _payload() + "\n```", LABELS)
    chunk = RetrievedChunk(
        chunk_id="c2", section_id="bns:303", act="BNS_2023", text="t", heading="Theft", score=0.5
    )
    claims, citations, passages = _resolve(parsed, {"E1": chunk, "E2": chunk})
    assert claims[0].evidence_labels == ["E2"]
    assert [c.section_id for c in citations] == ["bns:303"]
    assert passages == [chunk]
