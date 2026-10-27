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
import math
import re
from dataclasses import dataclass
from datetime import datetime
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


_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_MONTH_DATE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+"
    r"\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b",
    re.IGNORECASE,
)
# "12-1 momentum" is a label, not a quantity. Mask it before number checks so
# the guard does not demand that both label components appear in the fact set.
_LABEL = re.compile(r"(?<![\w.])\d{1,3}-\d{1,3}(?![\w.%])")
# Numeric forms seen in prose: 1,000, 12.5, .5, -.5, +3, 1e-3, 1E+3.
_SIGN = r"[+\-−–]?"
_NUMBER = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+"
_NUMBER_GROUP = rf"{_SIGN}(?:{_NUMBER})"
_RATIO = re.compile(
    rf"(?<![\w.])({_NUMBER_GROUP})\s*([/:])\s*({_NUMBER_GROUP})"
    r"(?!\w)(?!\.\d)"
)
_PERCENT = re.compile(rf"(?<![\w.])({_NUMBER_GROUP})\s*%")
_UNIT = re.compile(
    rf"(?<![\w.])({_NUMBER_GROUP})\s*"
    r"(bps?|basis points?|x|×|yrs?|years?|days?|months?|weeks?|sd|σ)\b",
    re.IGNORECASE,
)
_CURRENCY = re.compile(rf"[$€£]\s*({_NUMBER_GROUP})")
_SCIENTIFIC = re.compile(
    rf"(?<![\w.])({_SIGN}\d+(?:\.\d+)?[eE][+\-]?\d+)(?!\w)"
)
# A sentence-final period is punctuation, not part of a malformed number.
# Matching the leading decimal in "1.2.3" is intentional: a dotted triple is
# not a financial quantity, so the first component must still be grounded.
_DECIMAL = re.compile(rf"(?<![\w.])({_SIGN}(?:\d+\.\d+|\.\d+))(?!\w)")
_INTEGER = re.compile(rf"(?<![\w.+\-−–])({_SIGN}\d+)(?!\w)(?!\.\d)")
# A dotted triple is not a financial quantity. Reject it rather than checking
# only its leading component.
_DOTTED = re.compile(r"(?<![\w.])[+\-−–]?\d+(?:\.\d+){2,}(?!\w)")


def guard_numbers(text: str, allowed: set[str]) -> None:
    """Reject prose containing financial numbers absent from the fact set.

    The guard covers decimals, percentages, basis points, quantities with
    units, currency amounts, ratios and dates. Bare integers are checked too,
    because an invented "Sharpe 3" is just as misleading as "Sharpe 3.14".
    Structural labels such as "12-1 momentum" are masked, and ISO/month-name
    dates must resolve to the decision date or another allowed date.
    """
    allowed_normalised = {_normalise(token) for token in allowed}
    allowed_dates = {
        token for token in allowed_normalised if _ISO_DATE.fullmatch(token)
    }

    for match in _ISO_DATE.findall(text):
        if match not in allowed_dates:
            raise GroundingError(f"unsupported date in generated text: {match}")
    text = _ISO_DATE.sub(" ", text)

    for match in _MONTH_DATE.findall(text):
        iso = _month_date_to_iso(match)
        if iso is None or iso not in allowed_dates:
            raise GroundingError(f"unsupported date in generated text: {match}")
    text = _MONTH_DATE.sub(" ", text)

    text = _LABEL.sub(" ", text)

    for match in _DOTTED.finditer(text):
        raise GroundingError(f"unsupported dotted number in generated text: {match.group(0)}")

    for match in _RATIO.finditer(text):
        token = f"{match.group(1)}/{match.group(3)}"
        if token not in allowed_normalised:
            raise GroundingError(f"unsupported ratio in generated text: {match.group(0)}")
    text = _RATIO.sub(" ", text)

    for match in _PERCENT.finditer(text):
        _check_number(match.group(1), allowed_normalised, "percentage")
    text = _PERCENT.sub(" ", text)

    for match in _UNIT.finditer(text):
        _check_number(match.group(1), allowed_normalised, "quantity")
    text = _UNIT.sub(" ", text)

    for match in _CURRENCY.finditer(text):
        _check_number(match.group(1), allowed_normalised, "currency amount")
    text = _CURRENCY.sub(" ", text)

    for match in _SCIENTIFIC.finditer(text):
        _check_number(match.group(1), allowed_normalised, "scientific number")
    text = _SCIENTIFIC.sub(" ", text)

    for match in _DECIMAL.finditer(text):
        _check_number(match.group(1), allowed_normalised, "decimal")
    text = _DECIMAL.sub(" ", text)

    for match in _INTEGER.finditer(text):
        _check_number(match.group(1), allowed_normalised, "integer")


def _check_number(token: str, allowed: set[str], kind: str) -> None:
    if _normalise(token) not in allowed:
        raise GroundingError(f"unsupported {kind} in generated text: {token}")


def _month_date_to_iso(text: str) -> str | None:
    cleaned = re.sub(r"(\d)(st|nd|rd|th)", r"\1", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned.replace(".", "").replace(",", "")).strip()
    cleaned = re.sub(r"(?i)\bsept\b", "Sep", cleaned)
    for fmt in ("%B %d %Y", "%b %d %Y"):
        try:
            return datetime.strptime(cleaned, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _normalise(token: str) -> str:
    token = str(token).strip().replace("−", "-").replace("–", "-").replace(",", "")
    token = re.sub(
        r"(?i)\s*(bps?|basis points?|x|×|yrs?|years?|days?|months?|weeks?|sd|σ)$",
        "",
        token,
    ).strip()
    token = token.rstrip("%").strip().lstrip("$€£").strip()
    try:
        return f"{float(token):.4f}"
    except ValueError:
        return token


def build_narrator(settings: Settings) -> Narrator:
    config = llm_config(settings)
    return RemoteNarrator(config) if config else DeterministicNarrator()


def numeric_tokens(*values: float, decimals: int = 4) -> set[str]:
    """Representations an LLM may legitimately use for each value.

    The draft narrator formats returns as percentages, Sharpe ratios to two
    decimals and costs in basis points, so all of those representations are
    allowed when they are meaningful. Coarse representations that round a
    nonzero value to zero are omitted: a cited ``0.00001`` must not whitelist a
    bare ``0`` in the prose.
    """
    tokens: set[str] = set()
    for raw in values:
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        magnitude = abs(value)

        def add(token: str, *, source: float = value) -> None:
            if source != 0.0 and float(token) == 0.0:
                return
            tokens.add(token)

        # Raw decimal forms used by the deterministic draft.
        if magnitude >= 0.01:
            add(f"{value:.2f}")
            add(f"{magnitude:.2f}")
        if magnitude >= 0.001:
            add(f"{value:.3f}")
            add(f"{magnitude:.3f}")
        add(f"{value:.{decimals}f}")
        add(f"{magnitude:.{decimals}f}")

        # Percent forms: the draft uses 1-2 decimals. Only weights, returns,
        # volatilities and risk measures are plausibly expressed this way;
        # emitting 400% for a risk class of 4 would whitelist a misleading
        # number. The zero-skip in ``add`` removes forms that round a nonzero
        # value to zero.
        if magnitude <= 1.0:
            percent = value * 100.0
            percent_magnitude = abs(percent)
            add(f"{percent:.1f}")
            add(f"{percent:.2f}")
            add(f"{percent_magnitude:.1f}")
            add(f"{percent_magnitude:.2f}")
            add(f"{percent:.0f}")
            add(f"{percent_magnitude:.0f}")

        # Basis-point forms used for turnover costs. The caller passes the cost
        # as a return fraction (e.g. 1.08 bps is 0.000108).
        if magnitude <= 1.0:
            basis_points = value * 10_000.0
            bp_magnitude = abs(basis_points)
            add(f"{basis_points:.1f}")
            add(f"{basis_points:.2f}")
            add(f"{bp_magnitude:.1f}")
            add(f"{bp_magnitude:.2f}")
            add(f"{basis_points:.0f}")
            add(f"{bp_magnitude:.0f}")
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
