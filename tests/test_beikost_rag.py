import csv
import sqlite3
from pathlib import Path

from evaluation.topic_guideline import write_results_csv
from pipeline.core.models import PipelineState, Statement
from pipeline.steps.guideline_classification import GuidelineClassificationStep
from pipeline.steps.retrieve_guideline_facts_RAG import retrieve_chunks
from pipeline.steps.topic_claim_normalization import (
    BEIKOST_NON_RECOMMENDATION_LABEL,
    TopicClaimNormalizationStep,
    classify_beikost_claim,
)


def test_classify_beikost_claim_routes_explanatory_claim_to_non_recommendation():
    stmt = Statement(
        id=1,
        text="Lebensmittelallergien treten nicht zwangsläufig bei der ersten Gabe auf, sondern können sich über Zeit bilden",
        translated_text_de="Lebensmittelallergien treten nicht zwangsläufig bei der ersten Gabe auf, sondern können sich über Zeit bilden",
        translated_text_en="Food allergies do not necessarily appear at first exposure and can develop over time.",
    )
    claim_type, flags, reason = classify_beikost_claim(stmt)
    assert claim_type == "explanatory_non_recommendation"
    assert "non_recommendation" in flags
    assert reason


def test_topic_claim_normalization_step_routes_non_recommendation_claim():
    step = TopicClaimNormalizationStep({"topic": "beikost"})
    state = PipelineState(statements=[Statement(id=1, text="Lebensmittelallergien treten nicht zwangsläufig bei der ersten Gabe auf, sondern können sich über Zeit bilden")])
    result = step.execute(state)
    stmt = result.statements[0]
    assert stmt.claim_type == "explanatory_non_recommendation"
    assert stmt.guideline_label == BEIKOST_NON_RECOMMENDATION_LABEL
    assert stmt.retrieval_status == "skipped_non_recommendation"
    assert stmt.classification_status == "skipped_non_recommendation"


def test_guideline_classifier_uses_no_evidence_fallback_without_blank_label():
    step = GuidelineClassificationStep(
        {
            "topic": "beikost",
            "allowed_labels": [
                "entspricht Leitlinie",
                "widerspricht Leitlinien",
                "wird nicht in Leitlinien genannt",
                "entspricht teilweise Leitlinien",
                "keine Beikostempfehlung",
            ],
            "label_definitions": {},
            "model": "dummy-model",
        }
    )
    state = PipelineState(statements=[Statement(id=1, text="Allergene sollten häufig angeboten werden")])
    result = step.execute(state)
    stmt = result.statements[0]
    assert stmt.guideline_label == "wird nicht in Leitlinien genannt"
    assert stmt.classification_status == "fallback_no_evidence"
    assert stmt.fallback_label_used is True


def test_guideline_classifier_falls_back_after_invalid_llm_response():
    class FakeLLM:
        def call(self, **kwargs):
            return '{"label":"entspricht Leitlinie","citations":[]}'

    step = GuidelineClassificationStep(
        {
            "topic": "beikost",
            "allowed_labels": [
                "entspricht Leitlinie",
                "widerspricht Leitlinien",
                "wird nicht in Leitlinien genannt",
                "entspricht teilweise Leitlinien",
                "keine Beikostempfehlung",
            ],
            "label_definitions": {},
            "model": "dummy-model",
            "max_retries": 0,
        }
    )
    step._llm_service = FakeLLM()
    stmt = Statement(
        id=1,
        text="Allergene sollten früh angeboten werden",
        claim_type="recommendation_paraphrase",
        evidence=[],
    )
    stmt.evidence = [
        type("FakeEvidence", (), {
            "source_type": "RAG",
            "chunk_id": "doc:1",
            "source_path": "/tmp/doc.pdf",
            "pages": [1],
            "abstract": "Allergene Lebensmittel sollen mit Beginn der Beikost eingeführt werden.",
        })()
    ]
    result = step.execute(PipelineState(statements=[stmt]))
    out = result.statements[0]
    assert out.guideline_label == "wird nicht in Leitlinien genannt"
    assert out.classification_status == "fallback_after_invalid_response"
    assert out.fallback_label_used is True


def test_retrieve_chunks_filters_bibliography_and_fuses_queries(monkeypatch, tmp_path):
    import pipeline.steps.retrieve_guideline_facts_RAG as rag

    db_path = tmp_path / "test.sqlite"
    sqlite3.connect(db_path).close()

    class FakeModel:
        pass

    def fake_retrieve(con, query, **kwargs):
        return [
            {
                "chunk_id": "good",
                "dense_score": 0.72,
                "fused_score": 0.11,
                "page_start": 10,
                "page_end": 10,
                "chunk_type": "recommendation",
                "text": "Eier und andere allergene Lebensmittel sollen mit Beginn der Beikost eingeführt werden.",
                "source_path": "/tmp/guideline.pdf",
                "title": "Guide",
                "section_path": "Empfehlung",
                "parent_chunk_id": None,
            },
            {
                "chunk_id": "bib",
                "dense_score": 0.9,
                "fused_score": 0.2,
                "page_start": 90,
                "page_end": 90,
                "chunk_type": "evidence",
                "text": "http://example.org Zugriff 2020 The Lancet 2016 world health organization 2019",
                "source_path": "/tmp/guideline.pdf",
                "title": "Literatur",
                "section_path": "Literatur",
                "parent_chunk_id": None,
            },
        ]

    monkeypatch.setattr(rag, "_load_sentence_transformer", lambda model_name: FakeModel())
    monkeypatch.setattr(rag, "retrieve", fake_retrieve)
    monkeypatch.setattr(rag, "add_parent_context", lambda con, results: None)

    _, usable, raw = retrieve_chunks(
        db_path,
        "Allergene sollten früh angeboten werden",
        queries=[
            "Allergene sollten früh angeboten werden",
            "Die Leitlinie macht eine Empfehlung zur Einführung potenziell allergener Lebensmittel in der Beikost.",
        ],
        final_k=3,
    )
    assert len(raw) >= 2
    assert [chunk.chunk_id for chunk in usable] == ["good"]


def test_write_results_csv_persists_debugging_fields(tmp_path):
    csv_path = tmp_path / "predictions.csv"
    items = [
        {
            "topic": "beikost",
            "claim_id": 1,
            "video_id": "1",
            "url": "https://example.com",
            "source_row": "2",
            "source_claim_number": "1",
            "claim": "Claim",
            "gold_label": None,
            "predicted_label": "keine Beikostempfehlung",
            "claim_type": "explanatory_non_recommendation",
            "routing_reason": "Background claim",
            "retrieval_status": "skipped_non_recommendation",
            "classification_status": "skipped_non_recommendation",
            "failure_stage": "",
            "fallback_label_used": False,
            "retrieval_queries": [],
            "cited_chunk_ids": [],
            "evidence": [],
            "retrieved_chunk_count": 0,
            "usable_retrieved_chunk_count": 0,
            "raw_retrieved_chunk_count": 0,
            "vdb_path": "/tmp/test.sqlite",
            "vdb_sha256": "abc",
            "status": "ok",
            "error": None,
        }
    ]
    write_results_csv(csv_path, items)
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["predicted_label"] == "keine Beikostempfehlung"
    assert rows[0]["claim_type"] == "explanatory_non_recommendation"
    assert rows[0]["status"] == "ok"
    assert rows[0]["retrieved_chunk_count"] == "0"
