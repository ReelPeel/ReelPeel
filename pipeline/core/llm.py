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

            if self.observer:
                self.observer.on_artifact(
                    "LLM Prompt",
                    {
                        "model": model,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "stop": stop,
                        "prompt": prompt,
                    },
                    depth=0,
                )

            # Execute Call
            response = self.client.chat.completions.create(**kwargs)

            # --- TRACK USAGE ---
            if response.usage:
                u = response.usage
                self.token_usage["prompt_tokens"] += u.prompt_tokens
                self.token_usage["completion_tokens"] += u.completion_tokens
                self.token_usage["total_tokens"] += u.total_tokens

                # Log detailed usage artifact via observer
                if self.observer:
                    self.observer.on_artifact("LLM Usage Stats", {
                        "model": model,
                        "prompt": u.prompt_tokens,
                        "completion": u.completion_tokens,
                        "total": u.total_tokens
                    }, depth=0)  # Logger handles context

            content = response.choices[0].message.content
            return content.strip() if content else ""

        except Exception as e:
            raise RuntimeError(f"LLM Service Error [Model: {model}]: {e}") from e
