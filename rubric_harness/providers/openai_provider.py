"""OpenAI chat completions provider.

No key is required to import this module. The key is only read when the provider
is constructed, so an offline checkout stays importable and testable.
"""

from __future__ import annotations

import os

import requests

from .base import ProviderConfigurationError, ProviderRequestError

API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TIMEOUT = 60


class OpenAIProvider:
    """Calls the OpenAI chat completions endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: int = DEFAULT_TIMEOUT,
        name: str = "openai",
    ) -> None:
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ProviderConfigurationError(
                "The OpenAI provider needs an API key. Set the OPENAI_API_KEY "
                "environment variable, or pass api_key explicitly."
            )
        self.api_key = resolved_key
        self.model = model or os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
        self.timeout = timeout
        self._name = name
        self._calls = 0

    @property
    def name(self) -> str:
        """Short identifier used in reports."""
        return self._name

    @property
    def calls(self) -> int:
        """How many times this instance has been asked for a completion."""
        return self._calls

    def complete(self, prompt: str) -> str:
        """Send one prompt and return the message content."""
        self._calls += 1
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        try:
            response = requests.post(
                API_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise ProviderRequestError(f"OpenAI request failed: {exc}") from exc

        if response.status_code != 200:
            raise ProviderRequestError(
                f"OpenAI returned status {response.status_code}: {response.text[:500]}"
            )

        body = response.json()
        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderRequestError(
                f"Unexpected OpenAI response shape: {str(body)[:500]}"
            ) from exc
