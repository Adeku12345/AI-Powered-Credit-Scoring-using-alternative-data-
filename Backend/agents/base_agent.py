import os
import sys
import json

from backend.core.exceptions import CustomException
from backend.core.logger import logging


class BaseAgent:
    """
    Shared functionality for any agent that needs to call the Claude API:
    client setup, a plain-text call, and a strict-JSON call. Each concrete
    agent subclasses this and implements its own .run(...) method with its
    own prompt and own slice of responsibility.
    """

    name = "base_agent"

    def __init__(self):
        self._client = None

    def _get_client(self):
        try:
            import anthropic

            if self._client is None:
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if not api_key:
                    raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
                self._client = anthropic.Anthropic(api_key=api_key)
            return self._client

        except Exception as e:
            raise CustomException(e, sys)

    def call_llm(self, prompt: str, max_tokens: int = 500) -> str:
        try:
            logging.info(f"[{self.name}] calling LLM")
            client = self._get_client()
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(
                block.text for block in response.content if block.type == "text"
            ).strip()

        except Exception as e:
            raise CustomException(e, sys)

    def call_llm_json(self, prompt: str, max_tokens: int = 500) -> dict:
        try:
            raw_text = self.call_llm(prompt, max_tokens=max_tokens)

            if raw_text.startswith("```"):
                raw_text = raw_text.strip("`")
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
                raw_text = raw_text.strip()

            return json.loads(raw_text)

        except Exception as e:
            raise CustomException(e, sys)
