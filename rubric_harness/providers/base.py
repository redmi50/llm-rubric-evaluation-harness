"""Provider interfaces and shared errors."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


class ProviderConfigurationError(RuntimeError):
    """Raised when a provider cannot be constructed, usually a missing API key."""


class ProviderRequestError(RuntimeError):
    """Raised when a provider call fails."""


@runtime_checkable
class Provider(Protocol):
    """Anything that can turn a prompt into a completion."""

    @property
    def name(self) -> str:
        """Short identifier used in reports."""

    @property
    def calls(self) -> int:
        """How many times this instance has been asked for a completion."""

    def complete(self, prompt: str) -> str:
        """Return the completion text for the prompt."""
