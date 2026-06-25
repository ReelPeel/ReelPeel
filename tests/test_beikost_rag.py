import csv
import sqlite3

from evaluation.topic_guideline import write_results_csv
from pipeline.core.models import GuidelineDocumentResult, PipelineState, RAGEvidence, Statement
from pipeline.steps.guideline_classification import GuidelineClassificationStep
from pipeline.steps.retrieve_guideline_facts_RAG import retrieve_chunks
from pipeline.steps.topic_claim_normalization import TopicClaimNormalizationStep, classify_beikost_claim


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
    state = PipelineState(
        statements=[
            Statement(
                id=1,
                text="Lebensmittelallergien treten nicht zwangsläufig bei der ersten Gabe auf, sondern können sich über Zeit bilden",
            )
        ]
    )
    result = step.execute(state)
    stmt = result.statements[0]
    assert stmt.claim_type == "explanatory_non_recommendation"
    assert stmt.guideline_label == "wird nicht in Leitlinien genannt"
    assert stmt.retrieval_status == "skipped_non_recommendation"
    assert stmt.classification_status == "mapped_non_recommendation_to_not_mentioned"


def test_guideline_classifier_uses_no_evidence_fallback_without_blank_label():
    step = GuidelineClassificationStep(
        {
            "topic": "beikost",
            "allowed_labels": [
                "entspricht Leitlinie",
                "widerspricht Leitlinien",
                "wird nicht in Leitlinien genannt",
                "entspricht teilweise Leitlinien",
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


def test_guideline_classifier_labels_documents_then_unifies():
    class FakeLLM:
        def __init__(self):
            self.responses = iter(
                [
                    '{"label":"entspricht teilweise Leitlinien","citations":["C1"]}',
                    '{"label":"wird nicht in Leitlinien genannt","citations":[]}',
                    '{"label":"entspricht teilweise Leitlinien","documents":["D1"]}',
                ]
            )

        def call(self, **kwargs):
            return next(self.responses)

    step = GuidelineClassificationStep(
        {
            "topic": "beikost",
            "allowed_labels": [
                "entspricht Leitlinie",
                "widerspricht Leitlinien",
                "wird nicht in Leitlinien genannt",
                "entspricht teilweise Leitlinien",
            ],
            "label_definitions": {},
            "model": "dummy-model",
        }
    )
    step._llm_service = FakeLLM()
    stmt = Statement(
        id=1,
        text="Allergene sollten früh angeboten werden",
        claim_type="recommendation_paraphrase",
        guideline_documents=[
            GuidelineDocumentResult(
                document_id="doc-1",
                source_path="/tmp/doc-1.pdf",
                title="Doc 1",
                retrieved_chunk_count=1,
                evidence=[
                    RAGEvidence(
                        chunk_id="doc-1:chunk-1",
                        score=0.8,
                        source_path="/tmp/doc-1.pdf",
                        document_id="doc-1",
                        document_title="Doc 1",
                        pages=[1],
                        abstract="Die Leitlinie empfiehlt eine frühe, aber qualifizierte Einführung allergener Lebensmittel.",
                    )
                ],
            ),
            GuidelineDocumentResult(
                document_id="doc-2",
                source_path="/tmp/doc-2.pdf",
                title="Doc 2",
                retrieved_chunk_count=1,
                evidence=[
                    RAGEvidence(
                        chunk_id="doc-2:chunk-1",
                        score=0.5,
                        source_path="/tmp/doc-2.pdf",
                        document_id="doc-2",
                        document_title="Doc 2",
                        pages=[4],
                        abstract="Das Dokument behandelt andere Aspekte der Beikost, nicht diese konkrete Aussage.",
                    )
                ],
            ),
        ],
    )
    result = step.execute(PipelineState(statements=[stmt]))
    out = result.statements[0]
    assert [doc.label for doc in out.guideline_documents] == [
        "entspricht teilweise Leitlinien",
        "wird nicht in Leitlinien genannt",
    ]
    assert out.guideline_label == "entspricht teilweise Leitlinien"
    assert out.cited_chunk_ids == ["doc-1:chunk-1"]
    assert out.classification_status == "ok"
    assert out.fallback_label_used is False

def test_global_label_prefers_any_direct_match():
    class FakeLLM:
        def __init__(self):
            self.responses = iter(
                [
                    '{"label":"widerspricht Leitlinien","citations":["C1"]}',
                    '{"label":"entspricht Leitlinie","citations":["C1"]}',
                    '{"label":"wird nicht in Leitlinien genannt","citations":[]}',
                ]
            )

        def call(self, **kwargs):
            return next(self.responses)

    step = GuidelineClassificationStep(
        {
            "topic": "beikost",
            "allowed_labels": [
                "entspricht Leitlinie",
                "widerspricht Leitlinien",
                "wird nicht in Leitlinien genannt",
                "entspricht teilweise Leitlinien",
            ],
            "label_definitions": {},
            "model": "dummy-model",
        }
    )
    step._llm_service = FakeLLM()
    stmt = Statement(
        id=2,
        text="Test claim",
        guideline_documents=[
            GuidelineDocumentResult(
                document_id="doc-a",
                source_path="/tmp/doc-a.pdf",
                title="Doc A",
                retrieved_chunk_count=1,
                evidence=[
                    RAGEvidence(
                        chunk_id="doc-a:chunk-1",
                        score=0.7,
                        source_path="/tmp/doc-a.pdf",
                        document_id="doc-a",
                        document_title="Doc A",
                        pages=[1],
                        abstract="Contradictory evidence.",
                    )
                ],
            ),
            GuidelineDocumentResult(
                document_id="doc-b",
                source_path="/tmp/doc-b.pdf",
                title="Doc B",
                retrieved_chunk_count=1,
                evidence=[
                    RAGEvidence(
                        chunk_id="doc-b:chunk-1",
                        score=0.8,
                        source_path="/tmp/doc-b.pdf",
                        document_id="doc-b",
                        document_title="Doc B",
                        pages=[2],
                        abstract="Direct matching evidence.",
                    )
                ],
            ),
            GuidelineDocumentResult(
                document_id="doc-c",
                source_path="/tmp/doc-c.pdf",
                title="Doc C",
                retrieved_chunk_count=1,
                evidence=[
                    RAGEvidence(
                        chunk_id="doc-c:chunk-1",
                        score=0.3,
                        source_path="/tmp/doc-c.pdf",
                        document_id="doc-c",
                        document_title="Doc C",
                        pages=[3],
                        abstract="Unrelated evidence.",
                    )
                ],
            ),
        ],
    )
    out = step.execute(PipelineState(statements=[stmt])).statements[0]
    assert [doc.label for doc in out.guideline_documents] == [
        "widerspricht Leitlinien",
        "entspricht Leitlinie",
        "wird nicht in Leitlinien genannt",
    ]
    assert out.guideline_label == "entspricht Leitlinie"
    assert out.cited_chunk_ids == ["doc-b:chunk-1"]


def test_global_label_is_not_mentioned_only_when_all_documents_are_not_mentioned():
    class FakeLLM:
        def __init__(self):
            self.responses = iter(
                [
                    '{"label":"wird nicht in Leitlinien genannt","citations":[]}',
                    '{"label":"wird nicht in Leitlinien genannt","citations":[]}',
                ]
            )

        def call(self, **kwargs):
            return next(self.responses)

    step = GuidelineClassificationStep(
        {
            "topic": "beikost",
            "allowed_labels": [
                "entspricht Leitlinie",
                "widerspricht Leitlinien",
                "wird nicht in Leitlinien genannt",
                "entspricht teilweise Leitlinien",
            ],
            "label_definitions": {},
            "model": "dummy-model",
        }
    )
    step._llm_service = FakeLLM()
    stmt = Statement(
        id=3,
        text="Test claim",
        guideline_documents=[
            GuidelineDocumentResult(
                document_id="doc-a",
                source_path="/tmp/doc-a.pdf",
                title="Doc A",
                retrieved_chunk_count=1,
                evidence=[
                    RAGEvidence(
                        chunk_id="doc-a:chunk-1",
                        score=0.2,
                        source_path="/tmp/doc-a.pdf",
                        document_id="doc-a",
                        document_title="Doc A",
                        pages=[1],
                        abstract="No relevant mention.",
                    )
                ],
            ),
            GuidelineDocumentResult(
                document_id="doc-b",
                source_path="/tmp/doc-b.pdf",
                title="Doc B",
                retrieved_chunk_count=1,
                evidence=[
                    RAGEvidence(
                        chunk_id="doc-b:chunk-1",
                        score=0.2,
                        source_path="/tmp/doc-b.pdf",
                        document_id="doc-b",
                        document_title="Doc B",
                        pages=[1],
                        abstract="Also no relevant mention.",
                    )
                ],
            ),
        ],
    )
    out = step.execute(PipelineState(statements=[stmt])).statements[0]
    assert out.guideline_label == "wird nicht in Leitlinien genannt"
    assert out.cited_chunk_ids == []


def test_guideline_classifier_uses_fallback_when_document_response_is_invalid():
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
        guideline_documents=[
            GuidelineDocumentResult(
                document_id="doc-1",
                source_path="/tmp/doc.pdf",
                title="Doc",
                retrieved_chunk_count=1,
                evidence=[
                    RAGEvidence(
                        chunk_id="doc-1:chunk-1",
                        score=0.8,
                        source_path="/tmp/doc.pdf",
                        document_id="doc-1",
                        document_title="Doc",
                        pages=[1],
                        abstract="Allergene Lebensmittel sollen mit Beginn der Beikost eingeführt werden.",
                    )
                ],
            )
        ],
    )
    result = step.execute(PipelineState(statements=[stmt]))
    out = result.statements[0]
    assert out.guideline_documents[0].label == "wird nicht in Leitlinien genannt"
    assert out.guideline_documents[0].classification_status == "fallback_after_invalid_response"
    assert out.guideline_label == "wird nicht in Leitlinien genannt"
    assert out.classification_status == "ok_with_fallback"
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
                "doc_id": "doc-1",
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
                "doc_id": "doc-1",
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
    assert usable[0].document_id == "doc-1"


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
            "predicted_label": "wird nicht in Leitlinien genannt",
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
    assert rows[0]["predicted_label"] == "wird nicht in Leitlinien genannt"
    assert rows[0]["retrieved_chunk_count"] == "0"
    assert rows[0]["cited_chunks"] == "[]"
