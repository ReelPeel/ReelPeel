# ReelPeel

ReelPeel is a research prototype for examining health claims in short-form videos. It combines a config-driven medical fact-checking pipeline, a FastAPI backend, and a browser extension that overlays claims and evidence directly on Instagram Reels.

The repository supports two complementary workflows:

- a repeatable demo flow based on a bundled JSON response
- a live analysis flow that downloads a reel, transcribes it, retrieves biomedical evidence, and scores claim truthfulness

ReelPeel is for research and demonstration only. It is not medical advice, not a diagnostic system, and not a clinical decision-support tool.

## What Is Included

- `app/`: FastAPI backend for demo and live analysis endpoints
- `browser-extension/`: Manifest V3 extension for Instagram Reels
- `pipeline/`: config-driven extraction, retrieval, reranking, stance, and verification pipeline
- `services/`: PubMed proxy used by the pipeline
- `evaluation/`: evaluation scripts and datasets


## Core Architecture

For system descriptions, the paper-relevant runtime path is the live path built around `POST /process` plus `POST /evidence_summary`.

- Browser extension: a Manifest V3 service worker plus a content script that watches Reel pages, triggers backend analysis after a short scroll pause, renders a claim list, opens per-claim source views, and requests short evidence explanations on demand.
- FastAPI backend: `app/main.py` exposes the live analysis route, the evidence-summary route, and a separate fixed demo route.
- Config-driven pipeline: `app/pipeline.py` selects an audio or video config, then `PipelineOrchestrator` executes an ordered list of `PipelineStep` implementations over a shared `PipelineState`.
- Retrieval infrastructure: the pipeline uses a local PubMed proxy that rate-limits and caches NCBI requests in SQLite.
- Shared data contract: the backend returns a `PipelineState` containing `Statement` objects, and each statement carries typed evidence records (`PubMed`, optional guideline `RAG`, schema-level `Epistemonikos`).

## Demo Modes

| Mode | Endpoint | Purpose |
|---|---|---|
| Fixed demo | `POST /json` | Returns a bundled example result for stable, repeatable UI demos |
| Live processing | `POST /process` | Runs the full reel-to-evidence pipeline |
| Evidence context | `POST /evidence_summary` | Produces a short explanation for a single evidence item |

For architecture descriptions or papers, describe `POST /process` and `POST /evidence_summary`. `POST /json` is a bundled demo shortcut and does not execute the live retrieval and verification pipeline.

## System Requirements

- Linux environment recommended
- Conda or Mamba
- `ffmpeg` available on the system path
- Ollama or another OpenAI-compatible local endpoint
- Chrome or Edge for the browser extension
- Network access for live reel downloads, PubMed retrieval, and first-time Hugging Face model downloads

GPU acceleration is strongly recommended for live runs. CPU-only execution is possible but slow.

## Setup

Use the conda environment as the primary installation path:

```bash
conda env create -f environment.yml
conda activate factchecker
```

If you need a stricter rebuild of the original environment, inspect `environment.resolved.yml`. The checked-in `requirements.txt` is an environment snapshot, not the preferred public install path.

### Start the Local LLM Backend

Default pipeline configs expect Ollama on `http://localhost:11434/v1`.

```bash
export OLLAMA_CONTEXT_LENGTH=32768
ollama serve
```

Pull the default LLMs used by the app configs:

```bash
ollama pull gemma3:12b
ollama pull gemma3:27b
```

On first live run, the reranking and stance components may also download Hugging Face models referenced by the pipeline configuration.

## Start the Backend App

```bash
uvicorn app.main:app --host 0.0.0.0 --port 6006
```

Useful environment variables:

- `LLM_BASE_URL`: overrides the default `http://localhost:11434/v1`
- `LLM_API_KEY`: default is `ollama`
- `SUMMARY_MODEL`: model used by `/evidence_summary` (default: `gemma3:12b`)
- `SUMMARY_TEMPERATURE`
- `SUMMARY_MAX_TOKENS`

Quick health check:

```bash
curl http://localhost:6006/number
```

The pipeline will try to auto-start the PubMed proxy on demand. If that fails in your environment, run it manually:

```bash
python services/pubmed_proxy.py
```

## App Usage

### 1. Fixed Demo Response

This route returns a bundled analysis payload and is the easiest way to drive the UI without relying on live retrieval.

```bash
curl -X POST http://localhost:6006/json \
  -H "Content-Type: application/json" \
  -d '{"url":"https://www.instagram.com/reels/DIRM85ZifdM/"}'
```

Notes:

- the request body is accepted for interface compatibility
- the current implementation ignores the incoming URL and always returns the same bundled demo result
- this route does not execute the live retrieval and verification pipeline

### 2. Live Reel Processing

This route downloads a reel, extracts audio, transcribes it, retrieves PubMed evidence, and produces verdicts and scores.

```bash
curl -X POST http://localhost:6006/process \
  -H "Content-Type: application/json" \
  -d '{"url":"https://www.instagram.com/reels/C0hXZ3bNAbH/","mock":false}'
```

Notes:

- `mock: false` triggers the full live workflow
- temporary downloaded files are cleaned up after the request completes
- live mode depends on the configured models and external services being reachable

### 3. Local Mock Audio

`mock: true` expects a pre-recorded WAV file in the project root. The filename must match the reel id extracted from the URL.

Example:

- URL: `https://www.instagram.com/reels/DIRM85ZifdM/`
- expected file: `DIRM85ZifdM.wav`

```bash
curl -X POST http://localhost:6006/process \
  -H "Content-Type: application/json" \
  -d '{"url":"https://www.instagram.com/reels/DIRM85ZifdM/","mock":true}'
```

### 4. Evidence Context Summaries

This route turns one abstract into a short, claim-specific explanation.

```bash
curl -X POST http://localhost:6006/evidence_summary \
  -H "Content-Type: application/json" \
  -d '{
    "statement": "Early introduction of eggs prevents allergies in children.",
    "evidence": {
      "abstract": "Meta-analyses of randomized controlled trials have found that introducing eggs earlier during infancy reduced egg allergy risk."
    }
  }'
```

This endpoint still requires a reachable summary model such as `gemma3:12b`.

## Browser Extension

The extension adds a floating overlay on Instagram Reels and supports:

- claim discovery and triage
- source inspection per claim
- on-demand evidence context summaries
- a local Claim Vault stored in browser storage

The overlay is driven primarily by the structured `statements[]` response returned by the backend. In the committed build, the service worker calls the demo endpoint for repeatability, but the content script is organized around the live response shape: a list of statements, each with evidence items carrying titles, links, publication types, relevance scores, and stance signals.

### Configure the Backend Origin

Before loading the extension, set the backend origin in both files below.

1. In `browser-extension/background.js`, update:

```js
const API_BASE = "http://localhost:6006";
```

2. In `browser-extension/manifest.json`, update the same origin in:

- `host_permissions`
- `content_security_policy.extension_pages.connect-src`

The origin must match exactly. If you use `127.0.0.1`, use it consistently in both places.

### Install the Extension

1. Open `chrome://extensions`
2. Enable Developer Mode
3. Click `Load unpacked`
4. Select the `browser-extension/` directory

### Use the Extension

1. Open an Instagram Reel URL.
2. Pause scrolling briefly so the extension can queue analysis.
3. Wait until the floating overlay changes from `Finding checkable claims...`.
4. Press and hold the floating button to enter `Inquiry Mode`.
5. Select a claim to open its `Sources` view.
6. Click the `?` button next to a source to request claim-specific context.
7. Use `Pin` to store a claim in the local `Claim Vault`.
8. Open `Vault` to revisit pinned claims later in the same browser profile.

Important behavior in the current demo build:

- the extension waits about 20 seconds before requesting the fixed demo payload
- the request target is `POST /json`, not `POST /process`
- Claim Vault entries are stored locally through browser extension storage

### Switch the Extension to Live Processing

If you want the extension to call the live backend instead of the fixed demo payload, change `browser-extension/background.js`:

- replace `fetch(\`${API_BASE}/json\`, ...)` with `fetch(\`${API_BASE}/process\`, ...)`
- set `JSON_FETCH_DELAY_MS` to `0` or remove the delay logic entirely

## Pipeline at a Glance

The default live path is:

1. download reel or load local audio
2. convert video to audio when needed
3. transcribe speech with Whisper
4. extract medical claims with an LLM
5. generate PubMed queries
6. fetch and weight evidence
7. rerank relevance and estimate support or refute stance
8. assign verdicts and aggregate truth scores

Optional guideline retrieval is available through the RAG utilities in `pipeline/RAG_vdb/`.


### Guideline Vector Database

The local guideline index uses SQLite, normalized dense embeddings, FTS5 lexical search, parent context, and page-level provenance. First-time model use may download the selected Hugging Face model; indexing and querying make no application-level web calls.

Build or rebuild an index:

```bash
python pipeline/RAG_vdb/build_guideline_vdb.py \
  --pdf_dir ./guidelines \
  --db_path guidelines_vdb.sqlite \
  --embed_model NeuML/pubmedbert-base-embeddings \
  --chunk_tokens 220 \
  --overlap_tokens 40 \
  --parent_chunk_tokens 1000 \
  --parent_overlap_tokens 120
```

A database using the legacy schema can be intentionally replaced with `--force_reindex --reset_db`. Re-running normally skips unchanged documents; selecting another `--embed_model` adds only that model's missing embeddings.

Inspect corpus health:

```bash
python pipeline/RAG_vdb/inspect_guideline_vdb.py \
  --db_path guidelines_vdb.sqlite
```

Retrieve claim evidence with hybrid dense and BM25 ranking:

```bash
python pipeline/RAG_vdb/query_guideline_vdb.py \
  --db_path guidelines_vdb.sqlite \
  --query "Drug X is recommended for disease Y during pregnancy" \
  --claim_mode \
  --show_parent
```

Add optional cross-encoder reranking:

```bash
python pipeline/RAG_vdb/query_guideline_vdb.py \
  --db_path guidelines_vdb.sqlite \
  --query "Drug X is recommended for disease Y during pregnancy" \
  --claim_mode \
  --show_parent \
  --rerank_model cross-encoder/ms-marco-MiniLM-L-6-v2
```

Evaluate a JSONL relevance set containing `claim` plus `relevant_chunk_ids` and/or `relevant_pages`:

```bash
python pipeline/RAG_vdb/evaluate_retrieval.py \
  --db_path guidelines_vdb.sqlite \
  --eval_jsonl retrieval_eval.jsonl
```

## Technical Pipeline

The pipeline is config-driven. A `PipelineOrchestrator` receives a config with an ordered `steps` array, validates required models, ensures the PubMed proxy is available, instantiates each step through `StepFactory`, and executes everything over a shared `PipelineState`.

### Execution Model

- `PipelineState` is the shared state object passed through all steps
- `PipelineStep.run()` wraps each step with timing, token accounting, and execution logging
- `PipelineModule` allows nested step groups inside a single top-level config
- the final state is serialized as JSON for API responses and debugging artifacts

Minimal config shape:

```python
PIPELINE_CONFIG = {
    "name": "ExampleRun",
    "debug": True,
    "steps": [
        {"type": "mock_transcript", "settings": {"transcript_text": "..."}},
        {"type": "extraction", "settings": {"model": "gemma3:27b", "prompt_template": "..."}},
        {
            "type": "module",
            "settings": {
                "name": "Research",
                "steps": [
                    {"type": "generate_query", "settings": {...}},
                    {"type": "fetch_links", "settings": {"retmax": 10}},
                    {"type": "abstract_evidence", "settings": {}},
                    {"type": "weight_evidence", "settings": {"default_weight": 0.15}},
                ],
            },
        },
        {"type": "module", "settings": {"name": "Scores", "steps": [...] }},
        {"type": "truthness", "settings": {...}},
        {"type": "scoring", "settings": {"threshold": 0.4}},
    ],
}
```

### Step-by-Step Data Flow

1. Input acquisition
   `download_reel` fetches the target video for URL-based runs, `video_to_audio` extracts audio, and `audio_to_transcript` runs Whisper. For offline testing, `mock_transcript` or `mock_statements` can inject input directly.

2. Claim extraction
   `extraction` prompts an LLM to return a JSON list of claims. The step strips markdown fences, parses JSON, and maps claim strings into `Statement` objects. If parsing fails, it falls back to naive sentence splitting.

3. Query generation
   `generate_query` expands each statement into one or more PubMed boolean queries. The implementation deduplicates normalized queries and can run multiple prompt variants in parallel. The step also supports prefetching PubMed IDs during query generation.

4. Evidence retrieval
   `fetch_links` calls PubMed ESearch through the local proxy and creates or updates `PubMedEvidence` items for matching PMIDs. Query provenance is stored on the evidence objects.

5. Metadata enrichment and weighting
   `abstract_evidence` batch-fetches titles, abstracts, and publication types. `weight_evidence` then maps publication types to numeric evidence weights using regex-based rules with a default fallback.

6. Relevance and stance scoring
   `rerank_evidence` uses a cross-encoder reranker to score claim-evidence relevance and can drop low-relevance items using `min_relevance`. `stance_evidence` applies an NLI model to estimate `Supports`, `Refutes`, or `Neutral` probabilities for each remaining item.

7. Optional evidence filtering
   `filter_evidence` can run an LLM relevance gate over the enriched evidence set. This step exists in the framework but is disabled in the current FastAPI live configuration.

8. Verdict generation and aggregation
   `truthness` formats the evidence block and prompts an LLM to output `VERDICT` and `FINALSCORE` for each statement. `scoring` aggregates statement scores into `overall_truthiness`, currently up-weighting low scores below a threshold to penalize likely false or uncertain claims.

9. Optional guideline retrieval
   `retrieve_guideline_facts` can attach `RAG` evidence from the local SQLite vector database. Retrieved chunks are embedded into the same evidence flow as PubMed results. This step is available in the framework but is not enabled in the current FastAPI live configuration.

### Default Live App Configuration

The current FastAPI live path is built from `VIDEO_URL_PIPELINE_CONFIG` and uses these defaults:

- Whisper transcription: `tiny.en`
- claim extraction: `gemma3:27b`
- PubMed query generation: `gemma3:12b`
- reranking: `BAAI/bge-reranker-v2-m3`
- stance estimation: `cnut1648/biolinkbert-mednli`
- final verdict generation: `gemma3:27b`
- evidence thresholding after reranking: `min_relevance = 0.7`
- overall score aggregation threshold: `0.4`

### State and Evidence Schema

The main state object contains:

- `transcript`
- `audio_path`
- `video_path`
- `statements`
- `overall_truthiness`
- `generated_at`
- `execution_log`

Each `Statement` contains:

- `id`
- `text`
- `verdict`
- `rationale`
- `score`
- `queries`
- `queries_fetched`
- `evidence`

Evidence is a tagged union with three source types:

- `PubMed`: `pubmed_id`, `url`, `title`, `abstract`, `pub_type`, `weight`, `relevance`, `stance`
- `RAG`: `chunk_id`, `source_path`, `pages`, `abstract`, `score`, `weight`, `relevance`
- `Epistemonikos`: schema exists, but it is not wired into the current pipeline path

The nested `stance` object stores:

- `abstract_label`
- `abstract_p_supports`
- `abstract_p_refutes`
- `abstract_p_neutral`

### UI-Relevant Response Fields

The current overlay mainly consumes:

- `statements[].text` for claim display
- `statements[].evidence[]` for source inspection
- evidence `title`, `url`, `pub_type`, `relevance`, and `stance` for source metadata
- evidence `abstract` when the user requests a short explanation through `POST /evidence_summary`

`overall_truthiness`, `verdict`, and `score` are part of the backend response, but the committed UI is organized first around claim triage and evidence inspection rather than a single global score display.

## Output Shape

A successful analysis response contains:

- `transcript`
- `statements[]` with `id`, `text`, `verdict`, `score`, `queries`, and `evidence`
- `overall_truthiness`
- `execution_log`

Each evidence item may include:

- bibliographic metadata such as `pubmed_id`, `title`, `url`, and `pub_type`
- ranking and verification fields such as `weight`, `relevance`, and `stance`

## Running the Pipeline Without the Web App

For manual end-to-end runs outside the FastAPI app:

```bash
python pipeline/test.py
```

This uses one of the predefined configs from `pipeline/test_configs/`. In practice, you will usually adjust the selected config or its input paths before running it. The final structured result is written to `final_output.json`.

## Repository Layout

```text
app/                       FastAPI app and API routes
browser-extension/         Instagram extension overlay
pipeline/                  Core pipeline framework and step implementations
pipeline/test_configs/     Reference configs and prompts
services/                  PubMed proxy and service helpers
evaluation/                Evaluation scripts and datasets
docs/                      Notes and internal setup material
zzz_videos/                Video and subtitle artifacts
```

## Modularity and Current Scope

- The execution model is modular at the pipeline level: step ordering, model choices, prompt templates, and optional modules are declared in config rather than hardcoded in the orchestrator.
- LLM-backed stages such as extraction, query generation, evidence filtering, verdict generation, and evidence summarization are replaceable through config and prompt changes.
- The current prototype is still medically specialized. The checked-in prompts target medical claim extraction and PubMed query generation, the evidence weighting rules encode biomedical publication types, and the default ranking and stance models are biomedical models.
- PubMed retrieval is the only fully wired literature retrieval path in the current live system. Guideline RAG is implemented as an optional add-on, while `Epistemonikos` is present in the shared schema but not connected to a retrieval step in the default pipeline.
- Test and demo shortcuts such as `POST /json`, `mock: true`, `mock_transcript`, `mock_statements`, cached JSON outputs, and other offline conveniences are useful for demos and debugging but should not be treated as the core architecture.

## Troubleshooting

- If startup fails with `LLM Server Unreachable`, start Ollama or set `LLM_BASE_URL`.
- If a model is reported missing, pull it with `ollama pull <model-name>`.
- If live requests stall on evidence retrieval, start `python services/pubmed_proxy.py` manually.
- If the extension cannot reach the backend, re-check the origin in both `browser-extension/background.js` and `browser-extension/manifest.json`, then reload the extension.
- If no claims appear on Instagram, make sure you are on a Reel page rather than a general feed page.

## Limitations

- The committed browser extension build is tuned for stable demo playback, not live inference by default.
- Live processing depends on external services and model availability.
- Results are based on retrieved evidence and model outputs; they are not guaranteed to be clinically complete or correct.
