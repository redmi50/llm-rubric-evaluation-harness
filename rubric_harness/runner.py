"""Running the task bank against several providers, with a disk cache.

The cache key is a hash of the provider name and the prompt. A cached response is
served without calling the provider at all, which is what makes re-scoring cheap
and what makes the cache behaviour testable through the provider call counter.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ResponseRecord:
    """One model response to one task."""

    task_id: str
    provider_name: str
    prompt: str
    response_text: str
    cached: bool = False
    latency_ms: float = 0.0

    def as_dict(self) -> dict:
        """Serializable representation."""
        return {
            "task_id": self.task_id,
            "provider_name": self.provider_name,
            "prompt": self.prompt,
            "response_text": self.response_text,
            "cached": self.cached,
            "latency_ms": round(self.latency_ms, 3),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ResponseRecord":
        """Rebuild a record from a serialized mapping."""
        return cls(
            task_id=payload["task_id"],
            provider_name=payload["provider_name"],
            prompt=payload["prompt"],
            response_text=payload["response_text"],
            cached=bool(payload.get("cached", False)),
            latency_ms=float(payload.get("latency_ms", 0.0)),
        )


def cache_key(provider_name: str, prompt: str) -> str:
    """A stable cache key for a provider and prompt pair."""
    digest = hashlib.sha256(f"{provider_name}\x00{prompt}".encode("utf-8")).hexdigest()
    return digest


class ResponseCache:
    """A simple JSON file cache keyed on provider name and prompt."""

    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def path_for(self, provider_name: str, prompt: str) -> Path:
        """Filesystem location of the cache entry."""
        return self.directory / f"{cache_key(provider_name, prompt)}.json"

    def get(self, provider_name: str, prompt: str) -> str | None:
        """Return a cached response, or None on a miss."""
        path = self.path_for(provider_name, prompt)
        if not path.is_file():
            self.misses += 1
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            self.misses += 1
            return None
        if payload.get("provider_name") != provider_name or payload.get("prompt") != prompt:
            self.misses += 1
            return None
        self.hits += 1
        return str(payload["response_text"])

    def put(self, provider_name: str, prompt: str, response_text: str) -> None:
        """Store a response."""
        payload = {
            "provider_name": provider_name,
            "prompt": prompt,
            "response_text": response_text,
        }
        self.path_for(provider_name, prompt).write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )


def run_matrix(
    tasks,
    providers,
    cache_dir: Path | str | None = None,
    use_cache: bool = True,
) -> list[ResponseRecord]:
    """Run every task against every provider.

    Results are returned sorted by task id then provider name so that repeated
    runs and reports are directly comparable.
    """
    cache = ResponseCache(cache_dir) if (cache_dir and use_cache) else None
    records: list[ResponseRecord] = []

    for task in sorted(tasks, key=lambda item: item.task_id):
        for provider in sorted(providers, key=lambda item: item.name):
            cached_text = cache.get(provider.name, task.prompt) if cache else None
            if cached_text is not None:
                records.append(
                    ResponseRecord(
                        task_id=task.task_id,
                        provider_name=provider.name,
                        prompt=task.prompt,
                        response_text=cached_text,
                        cached=True,
                        latency_ms=0.0,
                    )
                )
                continue

            started = time.perf_counter()
            response_text = provider.complete(task.prompt)
            elapsed_ms = (time.perf_counter() - started) * 1000.0

            if cache is not None:
                cache.put(provider.name, task.prompt, response_text)

            records.append(
                ResponseRecord(
                    task_id=task.task_id,
                    provider_name=provider.name,
                    prompt=task.prompt,
                    response_text=response_text,
                    cached=False,
                    latency_ms=elapsed_ms,
                )
            )

    return records
