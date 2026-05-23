"""
Base LLM agent with resilient retry logic, latency tracking, and state logging.

All agents inherit from BaseAgent. It connects to OpenRouter's OpenAI-compatible
API endpoint, handles rate limits (429), server errors (5xx) via exponential
backoff, and logs every call to the SQLite StateManager.
"""

import os
import time
from openai import OpenAI

from src.core.state_manager import StateManager


class BaseAgent:
    """
    Shared LLM client used by Extractor, Critic, and Mutator.

    Retry strategy:
      - 429 (Rate Limit)  : exponential backoff starting at 15 s
      - 5xx (Server Error): fixed 20 s pause
      - Other exceptions  : re-raised immediately after logging
    """

    MAX_RETRIES = 5
    BASE_DELAY_429 = 15   # seconds; doubles each attempt
    DELAY_5XX = 20        # seconds; fixed
    INTER_CALL_PAUSE = 8  # seconds; polite pause between successful calls

    def __init__(self, model_name: str, state_manager: StateManager):
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENROUTER_API_KEY environment variable is not set. "
                "Export it before running: export OPENROUTER_API_KEY=your_key"
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        self.model_name = model_name
        self.state_manager = state_manager

    def call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        role_name: str,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> str:
        """
        Call the LLM with retry/backoff. Returns the assistant's text content.

        Parameters
        ----------
        system_prompt : Instruction context for the model.
        user_prompt   : User-facing input (document text, critique, etc.)
        role_name     : Label used in logs (e.g. "Extractor", "Critic").
        temperature   : Sampling temperature; lower = more deterministic.
        max_tokens    : Hard cap on response length.
        """
        last_exception = None

        for attempt in range(self.MAX_RETRIES):
            try:
                t_start = time.monotonic()
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                latency_ms = (time.monotonic() - t_start) * 1000
                content = response.choices[0].message.content or ""

                self.state_manager.log_llm_call(
                    role=role_name,
                    prompt=f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}",
                    response=content,
                    model=self.model_name,
                    usage=response.usage.model_dump() if response.usage else {},
                    cost=0.0,  # OpenRouter free tier
                    latency_ms=latency_ms,
                )

                time.sleep(self.INTER_CALL_PAUSE)
                return content

            except Exception as exc:
                last_exception = exc
                error_msg = str(exc)

                if attempt >= self.MAX_RETRIES - 1:
                    print(f"❌ [{role_name}] Max retries ({self.MAX_RETRIES}) reached. Failing.")
                    raise exc

                if "429" in error_msg:
                    delay = self.BASE_DELAY_429 * (2 ** attempt)
                    print(f"⚠️  [{role_name}] Rate-limited (429). Retrying in {delay}s "
                          f"(attempt {attempt + 1}/{self.MAX_RETRIES})…")
                    time.sleep(delay)
                elif any(code in error_msg for code in ("500", "502", "503")):
                    print(f"⚠️  [{role_name}] Server error. Retrying in {self.DELAY_5XX}s "
                          f"(attempt {attempt + 1}/{self.MAX_RETRIES})…")
                    time.sleep(self.DELAY_5XX)
                else:
                    raise exc  # Non-retryable error

        raise last_exception  # Should not reach here