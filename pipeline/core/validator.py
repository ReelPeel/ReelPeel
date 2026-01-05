import os
import re
from typing import Dict, Any, Set, Tuple, List

import httpx


def validate_pipeline_models(config: Dict[str, Any]):
    """
    Validates all models defined in the pipeline config.
    - 'model' keys are checked against the Ollama server.
    - 'model_name' keys are checked against the Hugging Face Hub (or local paths).
    """
    print("--- Validating Model Availability ---")

    # 1. Collect all models, separated by type
    llm_models, hf_models = _collect_models_recursive(config)

    errors = []

    # 2. Validate Ollama / LLM Models
    if llm_models:
        llm_errors = _validate_ollama_models(config, llm_models)
        errors.extend(llm_errors)
    else:
        print("    No LLM models ('model') found to validate.")

    # 3. Validate Hugging Face Models
    if hf_models:
        hf_errors = _validate_hf_models(hf_models)
        errors.extend(hf_errors)
    else:
        print("    No HF models ('model_name') found to validate.")

    # 4. Final Verdict
    if errors:
        print("\n[CRITICAL] MODEL VALIDATION FAILED")
        for err in errors:
            print(f"   - {err}")
        print("-" * 40)
        raise ValueError(f"Pipeline cannot start due to missing models.")

    print("[OK] All models validated successfully.")


def _collect_models_recursive(data: Any) -> Tuple[Set[str], Set[str]]:
    """
    Recursively finds all values for:
      - 'model': treated as LLM/Ollama models
      - 'model_name': treated as Hugging Face models
    Returns: (llm_models, hf_models)
    """
    llm_models = set()
    hf_models = set()

    if isinstance(data, dict):
        for k, v in data.items():
            if k == "model" and isinstance(v, str):
                llm_models.add(v)
            elif k == "model_name" and isinstance(v, str):
                hf_models.add(v)
            else:
                l, h = _collect_models_recursive(v)
                llm_models.update(l)
                hf_models.update(h)

    elif isinstance(data, list):
        for item in data:
            l, h = _collect_models_recursive(item)
            llm_models.update(l)
            hf_models.update(h)

    return llm_models, hf_models


def _validate_ollama_models(config: Dict, models: Set[str]) -> List[str]:
    """Checks if models exist on the Ollama server."""
    llm_settings = config.get("llm_settings", {})
    base_url = llm_settings.get("base_url", "http://localhost:11434/v1")

    # Clean URL for the /models endpoint
    # OpenAI API: /v1/models
    # Ollama Raw: /api/tags

    available_models = set()
    missing = []

    try:
        # Try OpenAI-compatible endpoint first
        models_url = f"{base_url.rstrip('/')}/models"
        # print(f"    Querying LLM Server: {models_url}...")

        with httpx.Client(timeout=5.0) as client:
            resp = client.get(models_url)

            if resp.status_code == 404:
                # Fallback to Ollama native API if /v1/models is missing
                alt_url = re.sub(r"/v1$", "", base_url.rstrip('/')) + "/api/tags"
                # print(f"    Retrying with Ollama Native: {alt_url}...")
                resp = client.get(alt_url)
                resp.raise_for_status()
                data = resp.json()
                # Ollama native returns 'name'
                available_models = {m["name"] for m in data.get("models", [])}
            else:
                resp.raise_for_status()
                data = resp.json()
                # OpenAI returns 'id'
                available_models = {m["id"] for m in data.get("data", [])}

    except Exception as e:
        print(f"    [WARNING] Could not connect to LLM server: {e}")
        # If we can't check, we warn but don't crash the pipeline unless strict
        return [f"LLM Server Unreachable: {e}"]

    for req in models:
        # Check exact match or :latest match
        if req not in available_models and f"{req}:latest" not in available_models:
            print(f"    [ERROR] LLM Model missing: {req}")
            missing.append(f"Missing LLM: {req} (Run: 'ollama pull {req}')")
        else:
            print(f"    [OK] LLM: {req}")

    return missing


def _validate_hf_models(models: Set[str]) -> List[str]:
    """Checks if models exist on the Hugging Face Hub (or are local files)."""
    missing = []

    with httpx.Client(timeout=10.0) as client:
        for m in models:
            # 1. Check if it's a local path first
            if os.path.isdir(m) or os.path.isfile(m):
                print(f"    [OK] HF (Local): {m}")
                continue

            # 2. Check Hugging Face Hub API
            # API: https://huggingface.co/api/models/{repo_id}
            url = f"https://huggingface.co/api/models/{m}"

            try:
                resp = client.get(url)
                if resp.status_code == 200:
                    print(f"    [OK] HF (Hub): {m}")
                elif resp.status_code == 401 or resp.status_code == 403:
                    print(f"    [WARN] HF (Private/Gated): {m} (Assuming valid)")
                else:
                    print(f"    [ERROR] HF Model not found: {m}")
                    missing.append(f"Missing HF Model: {m}")
            except Exception as e:
                print(f"    [WARN] Could not check HF Hub for {m}: {e}")
                # Don't fail on network errors (might be offline mode)

    return missing
