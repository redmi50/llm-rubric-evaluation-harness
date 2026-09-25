"""Provider registry.

Resolves a provider name into a provider instance. Mock profiles work offline.
The OpenAI and Anthropic providers are only constructed when the relevant API key
is present in the environment, and they fail with a message that names the exact
variable to set rather than a generic error.
"""

from __future__ import annotations

from .base import Provider, ProviderConfigurationError, ProviderRequestError
from .mock import PROFILES, DeterministicMockProvider

DEFAULT_MOCK_PROFILE = "exact"


def build_reference_table(tasks) -> dict[str, str]:
    """Map every task prompt to its reference answer."""
    return {task.prompt: task.expected_reference for task in tasks}


def get_provider(name: str, tasks=(), **kwargs) -> Provider:
    """Resolve a provider by name.

    Accepted names:
        ``mock`` or ``mock-exact`` and the other mock profiles
        ``openai``
        ``anthropic``
    """
    answers = build_reference_table(tasks) if tasks else {}

    if name == "mock":
        return DeterministicMockProvider(DEFAULT_MOCK_PROFILE, answers)
    if name.startswith("mock-"):
        profile = name[len("mock-") :]
        if profile not in PROFILES:
            raise ValueError(
                f"Unknown mock profile {profile!r}. Expected one of {PROFILES}."
            )
        return DeterministicMockProvider(profile, answers)

    if name == "openai":
        from .openai_provider import OpenAIProvider

        return OpenAIProvider(**kwargs)

    if name == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(**kwargs)

    raise ValueError(
        f"Unknown provider {name!r}. Use mock, mock-<profile>, openai or anthropic."
    )


build_provider = get_provider

__all__ = [
    "Provider",
    "ProviderConfigurationError",
    "ProviderRequestError",
    "DeterministicMockProvider",
    "get_provider",
    "build_provider",
    "build_reference_table",
]
