import openai
from typing import Dict, Any, List, Optional, Union
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionUserMessageParam


class LLMService:
    def __init__(self, config: Dict[str, Any]):
        self.base_url = config.get("base_url", "http://localhost:11434/v1")
        self.api_key = config.get("api_key", "ollama")
        self.client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key)

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
        if temperature is None or temperature <= 0:
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

            content = response.choices[0].message.content
            return content.strip() if content else ""

        except Exception as e:
            raise RuntimeError(f"LLM Service Error [Model: {model}]: {e}") from e
