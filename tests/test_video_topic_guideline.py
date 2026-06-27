import csv
import json

from evaluation.topic_guideline import TOPIC_PROFILES
from evaluation.video_topic_guideline import (
    build_video_pipeline_config,
    load_video_transcripts,
    run_video_topic_evaluation,
    transcript_path_for_topic,
)
from pipeline.core.models import PipelineState, Statement
from pipeline.steps.extraction import TranscriptToStatementStep


def _write_transcript_file(root, filename, items):
    path = root / filename
    path.write_text(
        json.dumps(
            {
                "generated_at": "2026-01-01T00:00:00Z",
                "model": "large-v3",
                "device": "cuda:0",
                "root_folder": "/tmp/videos",
                "count_ok": len(items),
                "count_failed": 0,
                "items": items,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_load_video_transcripts_reads_expected_topic_file_and_preserves_empty_text(tmp_path):
    _write_transcript_file(
        tmp_path,
        "downloads_beikost.json",
        [
            {"file": "a.mp4", "language": "de", "text": "Ein Baby kann Beikost essen."},
            {"file": "b.mp4", "language": "en", "text": ""},
        ],
    )

    records = load_video_transcripts("beikost", tmp_path)

    assert [record.video_file for record in records] == ["a.mp4", "b.mp4"]
    assert records[0].topic == "beikost"
    assert records[1].transcript == ""
    assert transcript_path_for_topic("beikost", tmp_path).name == "downloads_beikost.json"


def test_load_video_transcripts_rejects_malformed_items(tmp_path):
    _write_transcript_file(tmp_path, "downloads_vitamind.json", [{"language": "de", "text": "Text"}])

    try:
        load_video_transcripts("vitamin-d", tmp_path)
    except ValueError as exc:
        assert "missing 'file'" in str(exc)
    else:
        raise AssertionError("Expected malformed transcript item to fail")


def test_extraction_step_enforces_max_statements_cap():
    class FakeLLM:
        token_usage = {"total_tokens": 0}

        def call(self, **kwargs):
            return json.dumps([f"Claim {i}" for i in range(25)])

    step = TranscriptToStatementStep(
        {
            "model": "dummy",
            "prompt_template": "{transcript}",
            "temperature": 0.0,
            "max_statements": 20,
        }
    )
    step._llm_service = FakeLLM()

    state = step.execute(PipelineState(transcript="Transcript with many claims."))

    assert len(state.statements) == 20
    assert state.statements[0].text == "Claim 0"
    assert state.statements[-1].text == "Claim 19"


def test_video_pipeline_config_routes_topics_to_expected_vdbs():
    beikost = build_video_pipeline_config(
        TOPIC_PROFILES["beikost"], model="dummy", top_k=7, min_score=0.2
    )
    vitamin = build_video_pipeline_config(
        TOPIC_PROFILES["vitamin-d"], model="dummy", top_k=7, min_score=0.2
    )

    assert beikost["steps"][3]["settings"]["topic"] == "beikost"
    assert "beikost_vdb.sqlite" in beikost["steps"][4]["settings"]["db_path"]
    assert vitamin["steps"][3]["settings"]["topic"] == "vitamin-d"
    assert "vitaminD_vdb.sqlite" in vitamin["steps"][4]["settings"]["db_path"]


def test_run_video_topic_evaluation_writes_statement_rows_and_checkpoint(monkeypatch, tmp_path):
    _write_transcript_file(
        tmp_path,
        "downloads_beikost.json",
        [{"file": "video.mp4", "language": "de", "text": "Transcript text."}],
    )

    import evaluation.video_topic_guideline as video_eval

    monkeypatch.setattr(
        video_eval,
        "validate_vdb",
        lambda profile: {
            "path": str(tmp_path / "fake.sqlite"),
            "sha256": "vdb-sha",
            "documents": 1,
            "chunks": 1,
            "embed_model": "model",
            "dimension": 384,
        },
    )

    class FakeOrchestrator:
        seen_configs = []

        def __init__(self, config):
            self.config = config
            self.seen_configs.append(config)

        def run(self, state):
            return PipelineState(
                statements=[
                    Statement(
                        id=1,
                        text="Statement one",
                        guideline_label="entspricht Leitlinie",
                        retrieval_status="ok",
                        classification_status="ok",
                    ),
                    Statement(
                        id=2,
                        text="Statement two",
                        guideline_label="wird nicht in Leitlinien genannt",
                        retrieval_status="skipped_non_recommendation",
                        classification_status="mapped_non_recommendation_to_not_mentioned",
                    ),
                ]
            )

    monkeypatch.setattr(video_eval, "PipelineOrchestrator", FakeOrchestrator)

    outputs = run_video_topic_evaluation(
        TOPIC_PROFILES["beikost"],
        output_root=tmp_path / "out",
        model="dummy",
        top_k=3,
        min_score=0.25,
        transcripts_root=tmp_path,
    )

    checkpoint = json.loads(outputs["checkpoint"].read_text(encoding="utf-8"))
    assert sorted(checkpoint["items"]) == ["video.mp4::1", "video.mp4::2"]
    assert FakeOrchestrator.seen_configs[0]["steps"][0]["settings"]["transcript_text"] == "Transcript text."

    with outputs["csv"].open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["video_statement_id"] for row in rows] == ["1", "2"]
    assert rows[0]["video_file"] == "video.mp4"
    assert rows[0]["statement"] == "Statement one"
    assert rows[0]["predicted_label"] == "1"
    assert rows[0]["predicted_label_text"] == "entspricht Leitlinie"
    assert rows[1]["predicted_label"] == "3"
    assert rows[1]["predicted_label_text"] == "wird nicht in Leitlinien genannt"
