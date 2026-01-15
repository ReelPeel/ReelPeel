# ReelPeel Medical Fact-Checking Pipeline

End-to-end, config-driven pipeline for medical claim extraction, PubMed evidence retrieval, optional guideline RAG retrieval, and conservative truth scoring.

This project treats statement scores as truth scores: 0.0 = claim is false, 1.0 = claim is true (with uncertainty in between).

## Pipeline overview (config-driven)

The pipeline is defined by a config dict with an ordered list of steps. The `PipelineOrchestrator` builds each step and runs them sequentially. A step can be a simple operation or a `module` that groups nested steps (still executed in order). The exact ordering and which steps are enabled is controlled by the config; the outline below describes the typical flow used in this repo.

## Detailed pipeline flow (typical order)

1. Input ingestion
   - A step populates `PipelineState.transcript` (from ASR or a mock step), or directly populates `PipelineState.statements` (skipping extraction).
   - The pipeline core only requires transcript text or prebuilt statements; audio ingestion lives outside the pipeline.

2. Claim extraction (`extraction`)
   - LLM turns the transcript into 1-3 medical claims.
   - Output is parsed as JSON; if parsing fails the step falls back to sentence splitting.
   - Produces `Statement` objects with `id` and `text`.

3. Research module: query generation -> retrieval -> metadata -> weighting
   - One or more `generate_query` steps expand each claim into PubMed Boolean queries.
     - Queries are normalized (tags, operators, parentheses) and deduplicated per statement.
     - Multiple query strategies can be chained, including counter-evidence oriented prompts.
   - `fetch_links` calls PubMed ESearch via a local proxy to retrieve PMIDs per query.
     - Evidence items are created (or updated). When an existing PMID is seen again, the query is recorded for provenance.
   - `summarize_evidence` batch-fetches publication types (esummary) and title/abstract text (efetch).
     - Titles are stored when available; abstracts are stored as raw text.
   - `weight_evidence` converts publication types to numeric weights using regex rules with a default fallback.

4. Optional guideline RAG retrieval (`retrieve_guideline_facts`)
   - Loads a SQLite vector DB built from guideline PDFs.
   - Reads the embedding model recorded in the DB, encodes each claim, computes cosine similarity, and returns top-k chunks above `min_score`.
   - Adds `RAGEvidence` items with `score`, `relevance`, and `weight` set from the similarity.

5. Scores module: relevance reranking + stance inference
   - `rerank_evidence` uses a cross-encoder (BAAI/bge-reranker-v2-m3 by default) to score claim/evidence relevance.
     - Titles are prefixed to abstracts when available.
     - `min_relevance` can drop low-scoring evidence.
   - `stance_evidence` uses an NLI model (BioLinkBERT MedNLI by default) to estimate support/refute/neutral probabilities.
     - `top_m_by_relevance` can limit stance computation to the most relevant items.

6. Verification module: filter -> verdict -> aggregate
   - `filter_evidence` uses an LLM to drop off-topic evidence based on abstract text plus metadata (weight, relevance, stance).
     - Evidence without text is kept for manual review.
     - All evidence types are eligible for filtering if they include text.
   - `truthness` builds a grouped evidence block (PubMed, Epistemonikos, RAG) and asks an LLM for a verdict and score.
     - Evidence is sorted by relevance (lowest to highest) before formatting.
     - The transcript context is provided to the prompt.
   - `scoring` aggregates statement scores into `overall_truthiness` and up-weights scores below a threshold to penalize likely false/uncertain claims.

7. Output
   - `PipelineState` contains statements, evidence, verdicts, and scores.
   - The debug log and summary table reflect step timing and token usage.

## Evidence model

All evidence items share a union schema and live in `Statement.evidence`:

- PubMed evidence (`source_type = PubMed`)
  - `pubmed_id`, `url`, `title`, `abstract`, `pub_type`, `weight`, `relevance`, `stance`, `queries`
- Guideline RAG evidence (`source_type = RAG`)
  - `chunk_id`, `source_path`, `pages`, `abstract` (chunk text), `score`, `weight`, `relevance`
- Epistemonikos evidence (`source_type = Epistemonikos`)
  - Defined in the schema but not wired into the pipeline yet.

## Configuration

### Config structure and modules

A pipeline is defined as a dict (or JSON) with a top-level `steps` array. Each entry has a `type` and `settings`. Use the special `module` type to group nested steps.

```python
PIPELINE_CONFIG = {
    "name": "Example",
    "debug": True,
    "steps": [
        {"type": "mock_transcript", "settings": {"transcript_text": "..."}},
        {"type": "extraction", "settings": {...}},
        {
            "type": "module",
            "settings": {
                "name": "Research",
                "steps": [
                    {"type": "generate_query", "settings": {...}},
                    {"type": "fetch_links", "settings": {"retmax": 20}},
                    {"type": "summarize_evidence", "settings": {}},
                    {"type": "weight_evidence", "settings": {"default_weight": 0.5}},
                ],
            },
        },
        {"type": "retrieve_guideline_facts", "settings": {...}},
        {"type": "module", "settings": {"name": "Scores", "steps": [...] }},
        {"type": "module", "settings": {"name": "Verification", "steps": [...] }},
    ],
}
```

### Pipeline step types (reference)

Key step types registered in `pipeline/core/factory.py`:

- `mock_transcript` and `mock_statements` for test input injection.
- `video_to_audio` for video -> audio file conversion.
- `audio_to_transcript` for audio -> transcript (Whisper).
- `extraction` for transcript -> statements.
- `generate_query`, `fetch_links`, `summarize_evidence`, `weight_evidence` for PubMed research.
- `retrieve_guideline_facts` for guideline RAG.
- `rerank_evidence`, `stance_evidence` for scoring.
- `filter_evidence`, `truthness`, `scoring` for verification.

### Prompt templates

Prompt templates are centralized in `pipeline/test_configs/preprompts.py` and injected per step.

### LLM endpoint settings

Each LLM step can override endpoint settings with `llm_settings`:

```python
{
  "type": "extraction",
  "settings": {
    "model": "gemma3:27b",
    "prompt_template": PROMPT_TMPL_S2,
    "llm_settings": {
      "base_url": "http://localhost:11434/v1",
      "api_key": "ollama"
    }
  }
}
```

If omitted, defaults to `http://localhost:11434/v1`.

### Model validation

`PipelineOrchestrator` validates required models at startup:

- `model` keys are checked against the LLM endpoint (Ollama-compatible).
- `model_name` keys are checked against the Hugging Face cache or Hub.

The run fails fast if required models are missing.

### Relevance thresholding and ordering

- `rerank_evidence` supports `min_relevance`. Evidence below the threshold is dropped.
- `truthness` sorts evidence by relevance ascending before sending to the LLM.

## PubMed proxy service

The pipeline uses a local proxy (`services/pubmed_proxy.py`) to respect NCBI rate limits.
`PipelineOrchestrator` will auto-start it (and wait for `/health`) if it is not running.
You can also run it manually:

```bash
python services/pubmed_proxy.py
```

## Quick start (pipeline only)

1) Create an environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or with conda:

```bash
conda env create -f environment.yml
conda activate factchecker
```

2) Start an OpenAI-compatible LLM endpoint (default is Ollama):

```bash
ollama serve
ollama pull gemma3:27b
```

3) Run the pipeline runner:

```bash
python pipeline/test.py
```

The runner imports a reference config from `pipeline/test_configs`. Swap the import or call `PipelineOrchestrator` directly to run a different config.

## Guideline RAG setup

The RAG system uses a local SQLite vector DB and SentenceTransformers embeddings.

1) Build a guideline DB:

```bash
python pipeline/RAG_vdb/build_guideline_vdb.py \
  --pdf_dir /path/to/guidelines \
  --db_path pipeline/RAG_vdb/guidelines_vdb.sqlite
```

2) Enable RAG retrieval in a pipeline config:

```python
{
  "type": "retrieve_guideline_facts",
  "settings": {
    "db_path": "pipeline/RAG_vdb/guidelines_vdb.sqlite",
    "top_k": 5,
    "min_score": 0.25,
  },
}
```

3) Optional: use a minimal RAG-only config from `pipeline/test_configs` and pass it into `PipelineOrchestrator`.

## Output

Each statement includes:

- `verdict`: true | false | uncertain
- `score`: 0.00-1.00 (truth score)
- `evidence`: mixed list of PubMed + RAG evidence

The pipeline runner writes a full JSON snapshot to `final_output.json`.

## Repository layout

```
app/                       # FastAPI app and reel utilities (optional ingestion)
app/main.py                # API entry point
app/reel_utils.py          # Reel download and audio conversion
app/step_1_audio_to_transcript.py
pipeline/                  # Core pipeline framework
pipeline/core/             # Orchestrator, models, LLM client, logging
pipeline/steps/            # Extraction, research, scoring, verification, RAG
pipeline/RAG_vdb/          # Guideline RAG vector DB tools
pipeline/test.py           # Local pipeline runner (loads a reference config)
pipeline/test_configs/     # Reference configs and prompt templates
services/                  # PubMed proxy service
evaluation/                # Evaluation scripts and datasets
browser-extension/         # UI experiment
logs/                      # Debug logs (when enabled)
final_output.json          # Example output (generated by pipeline/test.py)
```

## Notes and caveats

- Evidence is abstract-only; summaries are not generated.
- Reranker and stance models prepend the paper title to the abstract when available.
- RAG chunk text is stored in the `abstract` field to integrate with the evidence schema.
- `filter_evidence` only keeps items with an abstract when the LLM replies with "yes"; errors default to dropping the item.
- Debug logs are saved as `logs/pipeline_debug_<run_id>.log` and prompt logs as `logs/pipeline_debug_<run_id>_prompts.log` when debug is enabled.
