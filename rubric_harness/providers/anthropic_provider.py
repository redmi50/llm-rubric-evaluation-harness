"""Anthropic messages provider.

No key is required to import this module. The key is only read when the provider
is constructed, so an offline checkout stays importable and testable.
"""

from __future__ import annotations

import os

import requests

from .base import ProviderConfigurationError, ProviderRequestError

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
DEFAULT_MODEL = "claude-3-5-sonnet-latest"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TIMEOUT = 60


class AnthropicProvider:
    """Calls the Anthropic messages endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: int = DEFAULT_TIMEOUT,
        name: str = "anthropic",
    ) -> None:
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ProviderConfigurationError(
                "The Anthropic provider needs an API key. Set the ANTHROPIC_API_KEY "
                "environment variable, or pass api_key explicitly."
            )
        self.api_key = resolved_key
        self.model = model or os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)
        self.max_tokens = max_tokens
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
        """Send one prompt and return the text content."""
        self._calls += 1
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        try:
            response = requests.post(
                API_URL,
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": API_VERSION,
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise ProviderRequestError(f"Anthropic request failed: {exc}") from exc

        if response.status_code != 200:
            raise ProviderRequestError(
                f"Anthropic returned status {response.status_code}: {response.text[:500]}"
            )

        body = response.json()
        try:
            blocks = body["content"]
            return "".join(
                block.get("text", "") for block in blocks if block.get("type") == "text"
            )
        except (KeyError, TypeError, AttributeError) as exc:
            raise ProviderRequestError(
                f"Unexpected Anthropic response shape: {str(body)[:500]}"
            ) from exc
