"""LLM access layer.

Two responsibilities:

1. ``complete_json`` — optional structured reasoning for analyst agents.
2. ``narrate`` — optional verbalisation of an already-computed recommendation.

Both are strictly optional. The deterministic narrator is the default and the
fallback. Output from a remote model passes a numeric-grounding guard: any
decimal or percentage that does not appear in the allowed set causes the text
to be rejected, because fabricated numbers are the failure mode the survey
identifies most consistently in financial LLM systems.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from intellivest.config import Settings


class LLMUnavailable(RuntimeError):
    """Raised when the configured provider cannot be reached or returns junk."""


class GroundingError(RuntimeError):
    """Raised when generated prose contains numbers not present in the facts."""


@dataclass(frozen=True)
class LLMConfig:
    model: str
    base_url: str
    api_key: str
    timeout_s: float


def llm_config(settings: Settings) -> LLMConfig | None:
    if not settings.llm_enabled:
        return None
    return LLMConfig(
        model=settings.llm_model,
        base_url=settings.llm_base_url.rstrip("/"),
        api_key=settings.llm_api_key or "",
        timeout_s=settings.llm_timeout_s,
    )


def _chat(config: LLMConfig, messages: list[dict[str, str]], *, json_mode: bool) -> str:
    body: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "temperature": 0.1,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    headers = {"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"}
    try:
        response = httpx.post(
            f"{config.base_url}/chat/completions",
            headers=headers,
            json=body,
            timeout=config.timeout_s,
        )
        response.raise_for_status()
        data = response.json()
        return str(data["choices"][0]["message"]["content"])
    except Exception as exc:  # network, auth, schema
        raise LLMUnavailable(str(exc)) from exc


def complete_json(config: LLMConfig, system: str, user: str) -> dict[str, Any]:
    content = _chat(
        config,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        json_mode=True,
    )
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise LLMUnavailable("provider did not return valid JSON") from exc
    if not isinstance(payload, dict):
        raise LLMUnavailable("provider returned a non-object JSON payload")
    return payload


class Narrator(Protocol):
    def narrate(self, system: str, prompt: str, *, allowed_numbers: set[str]) -> str: ...


class DeterministicNarrator:
    """Template narrator. Always available, always reproducible."""

    def narrate(self, system: str, prompt: str, *, allowed_numbers: set[str]) -> str:
        # The caller passes the finished paragraph as the prompt's tail.
        marker = "DRAFT:"
        return prompt.split(marker, 1)[1].strip() if marker in prompt else prompt.strip()


class RemoteNarrator:
    def __init__(self, config: LLMConfig, fallback: Narrator | None = None) -> None:
        self.config = config
        self.fallback = fallback or DeterministicNarrator()

    def narrate(self, system: str, prompt: str, *, allowed_numbers: set[str]) -> str:
        try:
            text = _chat(
                self.config,
                [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                json_mode=False,
            )
            guard_numbers(text, allowed_numbers)
            return text.strip()
        except (LLMUnavailable, GroundingError):
            return self.fallback.narrate(system, prompt, allowed_numbers=allowed_numbers)


_DECIMAL = re.compile(r"-?\d+\.\d+")
_PERCENT = re.compile(r"-?\d+(?:\.\d+)?\s*%")


def guard_numbers(text: str, allowed: set[str]) -> None:
    """Reject prose containing financial numbers absent from the fact set."""
    allowed_normalised = {_normalise(token) for token in allowed}
    for match in _DECIMAL.findall(text):
        if _normalise(match) not in allowed_normalised:
            raise GroundingError(f"unsupported decimal in generated text: {match}")
    for match in _PERCENT.findall(text):
        token = _normalise(match.replace("%", "").strip())
        if token not in allowed_normalised:
            raise GroundingError(f"unsupported percentage in generated text: {match}")


def _normalise(token: str) -> str:
    try:
        return f"{float(token):.4f}"
    except ValueError:
        return token


def build_narrator(settings: Settings) -> Narrator:
    config = llm_config(settings)
    return RemoteNarrator(config) if config else DeterministicNarrator()


def numeric_tokens(*values: float, decimals: int = 4) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        tokens.add(f"{value:.{decimals}f}")
        tokens.add(f"{value * 100:.1f}")
        tokens.add(f"{value * 100:.2f}")
        tokens.add(f"{abs(value) * 100:.1f}")
        tokens.add(f"{abs(value) * 100:.2f}")
    return tokens


__all__ = [
    "DeterministicNarrator",
    "GroundingError",
    "LLMConfig",
    "LLMUnavailable",
    "Narrator",
    "RemoteNarrator",
    "build_narrator",
    "complete_json",
    "guard_numbers",
    "llm_config",
    "numeric_tokens",
]
