import os
import uuid
import openai
from typing import Dict, Any, Optional, List, Union
from .logging import PipelineObserver


class LLMService:
    def __init__(self, config: Dict[str, Any], observer: Optional[PipelineObserver] = None):
        self.base_url = config.get("base_url", "http://localhost:11434/v1")
        self.api_key = config.get("api_key", "ollama")
        self.client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key)

        self.observer = observer

        # Accumulators
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }

        # Context length (Ollama-specific, applied via options.num_ctx if available)
        ctx_val = (
            config.get("context_length")
            or config.get("num_ctx")
            or config.get("max_context")
            or os.environ.get("LLM_CONTEXT_LENGTH")
            or os.environ.get("OLLAMA_CONTEXT_LENGTH")
        )
        self.context_length = None
        if ctx_val is None:
            base_url_lower = (self.base_url or "").lower()
            if "ollama" in base_url_lower or "11434" in base_url_lower:
                ctx_val = 65536
        if ctx_val is not None:
            try:
                ctx_val = int(ctx_val)
                if ctx_val > 0:
                    self.context_length = ctx_val
            except Exception:
                self.context_length = None

    def call(self,
             prompt: str,
             model: str,
             temperature: float,
             max_tokens: Optional[int] = None,
             stop: Optional[Union[str, List[str]]] = None
             ) -> str:

        # Validation
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if temperature is None or temperature < 0:
            raise ValueError("temperature must be a positive float")
        if self.context_length is not None and self.context_length <= 0:
            raise ValueError("context_length must be a positive integer")

        call_id = uuid.uuid4().hex

        try:
            # Prepare arguments
            kwargs = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
            }

            if stop is not None:
                kwargs["stop"] = stop
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            if self.context_length is not None:
                kwargs["extra_body"] = {
                    "options": {
                        "num_ctx": self.context_length
                    }
                }

            if self.observer:
                self.observer.on_artifact(
                    "LLM Prompt",
                    {
                        "call_id": call_id,
                        "model": model,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "context_length": self.context_length,
                        "stop": stop,
                        "prompt": prompt,
                    },
                    depth=0,
                )

            # Execute Call
            response = self.client.chat.completions.create(**kwargs)

            # --- TRACK USAGE ---
            usage_data = {
                "call_id": call_id,
                "model": model,
                "prompt": None,
                "completion": None,
                "total": None,
            }
            if response.usage:
                u = response.usage
                self.token_usage["prompt_tokens"] += u.prompt_tokens
                self.token_usage["completion_tokens"] += u.completion_tokens
                self.token_usage["total_tokens"] += u.total_tokens
                usage_data["prompt"] = u.prompt_tokens
                usage_data["completion"] = u.completion_tokens
                usage_data["total"] = u.total_tokens

            # Log detailed usage artifact via observer
            if self.observer:
                self.observer.on_artifact("LLM Usage Stats", usage_data, depth=0)

            content = response.choices[0].message.content
            return content.strip() if content else ""

        except Exception as e:
            if self.observer:
                self.observer.on_artifact(
                    "LLM Usage Stats",
                    {
                        "call_id": call_id,
                        "model": model,
                        "prompt": None,
                        "completion": None,
                        "total": None,
                        "error": str(e),
                    },
                    depth=0,
                )
            raise RuntimeError(f"LLM Service Error [Model: {model}]: {e}") from e
