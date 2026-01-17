# Copilot Instructions (ReelPeel)

## Big picture architecture
- The core is a **config-driven pipeline** that runs ordered steps via `PipelineOrchestrator` in pipeline/core/orchestrator.py; steps are created by `StepFactory` in pipeline/core/factory.py.
- Steps are `PipelineStep` subclasses (pipeline/core/base.py) that implement `execute()` and mutate `PipelineState` (pipeline/core/models.py). **Do not override `run()`**; it handles logging, timing, and token accounting.
- Typical flow: transcript → claim extraction → PubMed research → optional guideline RAG → rerank/stance → filter → LLM verdict → aggregate score. See README.md for the canonical step order and config shape.
- FastAPI entry point lives in app/main.py and delegates to app/pipeline.py, which builds configs from pipeline/test_configs (audio/video) and runs the orchestrator.

## Key modules & data contracts
- Data model is Pydantic: `PipelineState` holds transcript/audio/video paths, `Statement` list, and evidence (`PubMedEvidence`, `RAGEvidence`, `EpistemonikosEvidence`) with `source_type` discriminator. See pipeline/core/models.py.
- Evidence is enriched across steps (e.g., PubMed evidence created in pipeline/steps/research.py, relevance in pipeline/steps/rerank.py, stance in pipeline/steps/stance.py, verdict in pipeline/steps/verification.py).
- Guideline RAG loads a **SQLite vector DB** built from PDFs: pipeline/RAG_vdb/build_guideline_vdb.py creates the DB, retrieval is pipeline/steps/retrieve_guideline_facts_RAG.py.

## Critical workflows
- Run the pipeline locally (uses a reference config): `python pipeline/test.py` (writes final_output.json).
- Start LLM endpoint (defaults to Ollama-compatible, base_url `http://localhost:11434/v1`). The orchestrator validates all `model`/`model_name` entries at startup (pipeline/core/validator.py).
- PubMed proxy: `PipelineOrchestrator` auto-starts a local proxy (services/pubmed_proxy.py) if port 8080 is not healthy; logs go to a shared cache dir defined in pipeline/core/service_manager.py.
- Build guideline DB (for RAG): `python pipeline/RAG_vdb/build_guideline_vdb.py --pdf_dir ... --db_path pipeline/RAG_vdb/guidelines_vdb.sqlite`.

## Project-specific patterns & conventions
- Pipeline configs are dicts with `steps` arrays; use `type: "module"` to nest steps. See pipeline/test_configs/*.py and README.md for examples.
- LLM prompts are centralized in pipeline/test_configs/preprompts.py and passed via `prompt_template` in step configs.
- Evidence ordering: `truthness` sorts evidence by **ascending relevance** before building the LLM prompt (pipeline/steps/verification.py).
- PubMed query normalization and fallback behavior live in pipeline/steps/research.py; outputs are tracked via `stmt.queries` and evidence `queries` for provenance.
- Logging: enable `debug` in config to emit run logs and prompt logs under logs/ (pipeline/core/logging.py).

## Integration points & dependencies
- PubMed API calls go through the local proxy at `http://127.0.0.1:8080/proxy` (pipeline/steps/research.py).
- Reranker/stance steps require torch + transformers; models are cached in-process for reuse (pipeline/steps/rerank.py, pipeline/steps/stance.py).
- SentenceTransformers + sqlite3 are required for guideline RAG retrieval (pipeline/steps/retrieve_guideline_facts_RAG.py).

## Where to look for examples
- End-to-end runner: pipeline/test.py
- Configs/prompts: pipeline/test_configs/*, pipeline/test_configs/preprompts.py
- Step implementations: pipeline/steps/*.py
- FastAPI entry: app/main.py
- Evaluation in evaluation.py
