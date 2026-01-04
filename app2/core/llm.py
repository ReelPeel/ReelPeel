import json
from datetime import datetime

import openai
from typing import Dict, Any, List, Optional, Union
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionUserMessageParam


class LLMService:
    def __init__(self, config: Dict[str, Any], debug: bool = False, log_file: str = None):
        self.base_url = config.get("base_url", "http://localhost:11434/v1")
        self.api_key = config.get("api_key", "ollama")
        self.client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key)

        # Logging Config
        self.debug = debug
        self.log_file = log_file

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
             max_tokens: int,
             stop: Optional[Union[str, List[str]]] = None
             ) -> str:

        user_msg: ChatCompletionUserMessageParam = {"role": "user", "content": prompt}
        messages: List[ChatCompletionMessageParam] = [user_msg]

        if max_tokens is None or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if temperature is None or temperature < 0:
            raise ValueError("temperature must be a positive float")

        try:
            # Construct arguments dynamically to avoid sending None if not needed
            # (Though most libraries handle stop=None fine, this is safer)
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            if stop is not None:
                kwargs["stop"] = stop

            response = self.client.chat.completions.create(**kwargs)

            # --- 1. TRACK USAGE ---
            if response.usage:
                u = response.usage
                self.token_usage["prompt_tokens"] += u.prompt_tokens
                self.token_usage["completion_tokens"] += u.completion_tokens
                self.token_usage["total_tokens"] += u.total_tokens

                # --- 2. LOG PER-REQUEST USAGE ---
                self._log_usage_artifact(model, u)

            content = response.choices[0].message.content
            return content.strip() if content else ""

        except Exception as e:
            raise RuntimeError(f"LLM Service Error [Model: {model}]: {e}") from e

    def _log_usage_artifact(self, model: str, usage):
        """Logs the cost of a single call to the debug file."""
        if not self.debug or not self.log_file:
            return

        timestamp = datetime.now().strftime("%H:%M:%S")
        data = {
            "model": model,
            "prompt": usage.prompt_tokens,
            "completion": usage.completion_tokens,
            "total": usage.total_tokens
        }
        formatted = json.dumps(data, indent=2)

        # Visual indent
        prefix = "      "
        formatted = "\n".join([f"{prefix}{line}" for line in formatted.split("\n")])

        entry = (
            f"\n   >>> [ARTIFACT] LLMService @ {timestamp}\n"
            f"   LABEL: Usage Stats (Single Call)\n"
            f"   --------------------------------------------------\n"
            f"{formatted}\n"
            f"   --------------------------------------------------\n"
        )

        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(entry)
        except Exception:
            pass