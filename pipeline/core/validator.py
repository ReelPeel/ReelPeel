import os
import re
from typing import Dict, Any, Set, Tuple, List

import httpx

# Try to import huggingface_hub for robust checking
try:
    from huggingface_hub import try_to_load_from_cache, HfApi
    from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError

    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False


def validate_pipeline_models(config: Dict[str, Any]):
    """
    Validates all models defined in the pipeline config.
    - 'model' keys are checked against the Ollama server.
    - 'model_name' keys are checked using huggingface_hub (Strict Mode).
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

    available_models = set()
    missing = []

    try:
        models_url = f"{base_url.rstrip('/')}/models"

        with httpx.Client(timeout=5.0) as client:
            resp = client.get(models_url)

            if resp.status_code == 404:
                # Fallback to Ollama native API
                alt_url = re.sub(r"/v1$", "", base_url.rstrip('/')) + "/api/tags"
                resp = client.get(alt_url)
                resp.raise_for_status()
                data = resp.json()
                available_models = {m["name"] for m in data.get("models", [])}
            else:
                resp.raise_for_status()
                data = resp.json()
                available_models = {m["id"] for m in data.get("data", [])}

    except Exception as e:
        print(f"    [WARNING] Could not connect to LLM server: {e}")
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
    """Checks if models exist locally, in HF cache, or on the Hugging Face Hub."""
    missing = []

    # Initialize API if available
    api = HfApi() if HAS_HF_HUB else None

    for m in models:
        # 1. Check if it's a local path first
        if os.path.isdir(m) or os.path.isfile(m):
            print(f"    [OK] HF (Local Path): {m}")
            continue

        # 2. Check Hugging Face Cache (Fast)
        if HAS_HF_HUB:
            try:
                cached_path = try_to_load_from_cache(repo_id=m, filename="config.json")
                if cached_path:
                    print(f"    [OK] HF (Cache): {m}")
                    continue
            except Exception:
                pass

        # 3. Check Online (Robust via HfApi)
        if HAS_HF_HUB:
            try:
                # This throws specific errors for 404 vs 401/403
                api.model_info(repo_id=m)
                print(f"    [OK] HF (Online Hub): {m}")
            except RepositoryNotFoundError:
                print(f"    [ERROR] HF Model not found: {m}")
                missing.append(f"Model not found on HF Hub: {m}")
            except GatedRepoError:
                print(f"    [OK] HF (Gated/Private): {m} (Ensure you are logged in)")
            except Exception as e:
                # If network error or other API issue
                print(f"    [ERROR] HF Verification failed for {m}: {e}")
                missing.append(f"Could not verify HF model: {m}")

        # 4. Fallback if huggingface_hub is NOT installed (Raw HTTP)
        else:
            print(f"    [INFO] validating {m} via raw HTTP...")
            url = f"https://huggingface.co/api/models/{m}"
            with httpx.Client(timeout=10.0) as client:
                try:
                    resp = client.get(url)
                    if resp.status_code == 200:
                        print(f"    [OK] HF (Online Hub): {m}")
                    elif resp.status_code == 404:
                        print(f"    [ERROR] HF Model not found: {m}")
                        missing.append(f"Model not found: {m}")
                    else:
                        # Treat 403/401 as ERRORS in strict mode, because pipeline download will likely fail too
                        print(f"    [ERROR] HF Access Denied ({resp.status_code}): {m}")
                        missing.append(f"Access Denied for model: {m}")
                except Exception as e:
                    print(f"    [WARN] Network error checking {m}: {e}")
                    # Don't block pipeline on network error if using raw HTTP (risky but flexible)

    return missing