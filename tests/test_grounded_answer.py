from building_with_rag.contracts import RetrievedChunk
from building_with_rag.generation.answer import _resolve, split_claims


def test_split_claims_labels_and_bullets():
    claims = split_claims(
        "Intro:\n- Theft is punished [E2].\n- Other point. [E1, E3]\nNo label here."
    )
    assert [c.evidence_labels for c in claims] == [["E2"], ["E1", "E3"], []]
    assert claims[0].text == "Theft is punished ."


def test_resolve_skips_unknown_labels():
    chunk = RetrievedChunk(
        chunk_id="c2", section_id="bns:303", act="BNS_2023", text="t", heading="Theft", score=0.5
    )
    claims = split_claims("Theft punished [E2][E9].")
    citations, passages = _resolve(claims, {"E2": chunk})
    assert [c.section_id for c in citations] == ["bns:303"]
    assert passages == [chunk]
