"""Strict schema and grounding for the optional LLM analyst path.

The LLM never hands a free-form sentence to the coordinator. It must return a
JSON object that validates against the models below, and every signal must cite
the fact-sheet fields it used. Validation then:

* rejects fields that do not exist in the fact sheet,
* rejects any number in the summary or a rationale that is not represented by
  one of the cited field values,
* rejects LLM-supplied metrics or evidence, which are the easiest places to
  smuggle an invented Sharpe ratio or performance claim,
* rebuilds evidence from the cited fields so every claim is traceable.

Any failure raises :class:`LLMValidationError`; the calling agent falls back to
its deterministic analyst.
"""

from __future__ import annotations

import json
import re
from datetime import date
from typing import Any

from pydantic import Field, ValidationError, field_validator, model_validator

from intellivest.agents.llm import GroundingError, guard_numbers, numeric_tokens
from intellivest.contracts import (
    AgentReport,
    Direction,
    Evidence,
    Signal,
    StrictModel,
)
from intellivest.data.universe import TICKERS

_CITATION_PATH = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*$")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_VALID_SUBJECTS = set(TICKERS) | {"MARKET", "PORTFOLIO"}


class LLMValidationError(ValueError):
    """Raised when generated JSON is malformed, ungrounded or unsupported."""


class LLMSignalPayload(StrictModel):
    name: str = Field(min_length=1, max_length=80)
    subject: str = Field(min_length=1, max_length=32)
    direction: Direction
    strength: float = Field(ge=-1.0, le=1.0)
    rationale: str = Field(min_length=1, max_length=400)
    confidence: float = Field(ge=0.0, le=1.0, default=0.6)
    citations: list[str] = Field(min_length=1, max_length=16)

    @field_validator("citations")
    @classmethod
    def _citation_paths(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        for path in value:
            if len(path) > 120 or not _CITATION_PATH.fullmatch(path):
                raise ValueError(f"invalid citation path: {path!r}")
            if path not in cleaned:
                cleaned.append(path)
        return cleaned

    @model_validator(mode="after")
    def _direction_matches_strength(self) -> "LLMSignalPayload":
        if self.direction == "bullish" and self.strength < 0:
            raise ValueError("bullish signal cannot have negative strength")
        if self.direction == "bearish" and self.strength > 0:
            raise ValueError("bearish signal cannot have positive strength")
        if self.direction == "neutral" and abs(self.strength) > 0.15:
            raise ValueError("neutral signal must have |strength| <= 0.15")
        return self


class LLMReportPayload(StrictModel):
    summary: str = Field(min_length=1, max_length=1200)
    confidence: float = Field(ge=0.0, le=1.0)
    summary_citations: list[str] = Field(default_factory=list, max_length=32)
    signals: list[LLMSignalPayload] = Field(min_length=1, max_length=40)

    @field_validator("summary_citations")
    @classmethod
    def _summary_citation_paths(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        for path in value:
            if len(path) > 120 or not _CITATION_PATH.fullmatch(path):
                raise ValueError(f"invalid citation path: {path!r}")
            if path not in cleaned:
                cleaned.append(path)
        return cleaned


def build_prompt(role: str, agent: str, fact_sheet: dict[str, Any]) -> tuple[str, str]:
    """Return the system and user prompts for a structured analyst response."""
    schema = {
        "summary": "string, 1-1200 characters",
        "confidence": "number in [0, 1]",
        "summary_citations": ["dotted fact-sheet field paths used in the summary"],
        "signals": [
            {
                "name": "string, 1-80 characters",
                "subject": "one of the valid subjects",
                "direction": "bullish | bearish | neutral",
                "strength": "number in [-1, 1], consistent with direction",
                "rationale": "string, 1-400 characters",
                "confidence": "number in [0, 1]",
                "citations": ["one or more dotted fact-sheet field paths used by this signal"],
            }
        ],
    }
    system = (
        f"You are the {role} in a regulated investment-advisory pipeline. "
        "Return ONLY a JSON object that follows this shape:\n"
        f"{json.dumps(schema, indent=2)}\n"
        "Rules:\n"
        "- Use only the fact sheet in the user message. Do not invent data, "
        "prices, dates, performance or risk metrics.\n"
        "- Every signal must cite at least one exact dotted field path from the "
        "fact sheet, for example assets.SPX.momentum_12_1.\n"
        "- summary_citations must list every fact-sheet field used in the "
        "summary. Every number in the summary must appear in one of those "
        "fields.\n"
        "- Every number in a signal rationale must appear in that signal's "
        "cited field values. Do not include an evidence or metrics key; the "
        "pipeline derives evidence from citations.\n"
        "- direction must match the sign of strength. A neutral signal must "
        "have |strength| <= 0.15.\n"
        f"- Valid subjects: {', '.join(sorted(_VALID_SUBJECTS))}.\n"
        "- Return at least one signal. If the facts do not support a groundable "
        "signal, the response will be rejected and the deterministic analyst "
        "will be used.\n"
        "- Do not add explanations before or after the JSON."
    )
    user = json.dumps({"agent": agent, "fact_sheet": fact_sheet}, default=str)
    return system, user


def validate_llm_report(
    payload: dict[str, Any],
    *,
    agent: str,
    role: str,
    as_of: date,
    fact_sheet: dict[str, Any],
) -> AgentReport:
    """Validate a raw LLM payload and return a grounded report.

    Raises ``LLMValidationError`` on any schema, citation or numeric violation.
    """
    try:
        parsed = LLMReportPayload.model_validate(payload)
    except ValidationError as exc:
        raise LLMValidationError(f"schema violation: {exc}") from exc

    catalog = flatten_fact_sheet(fact_sheet)
    summary_values = _values_for(catalog, parsed.summary_citations)
    _guard(parsed.summary, summary_values)

    signals: list[Signal] = []
    cited_paths: list[str] = list(parsed.summary_citations)
    for item in parsed.signals:
        if item.subject not in _VALID_SUBJECTS:
            raise LLMValidationError(f"unknown signal subject: {item.subject!r}")
        values = _values_for(catalog, item.citations)
        _guard(item.rationale, values)
        try:
            signal = Signal(
                name=item.name,
                subject=item.subject,
                direction=item.direction,
                strength=item.strength,
                rationale=item.rationale,
                confidence=item.confidence,
                citations=item.citations,
            )
        except ValidationError as exc:
            raise LLMValidationError(f"signal violation: {exc}") from exc
        signals.append(signal)
        for path in item.citations:
            if path not in cited_paths:
                cited_paths.append(path)

    evidence = [
        _evidence_for(path, catalog[path], as_of)
        for path in cited_paths
    ]
    return AgentReport(
        agent=agent,
        role=role,
        as_of=as_of,
        summary=parsed.summary,
        confidence=parsed.confidence,
        signals=signals,
        evidence=evidence,
        metrics={},
        degraded=False,
    )


def flatten_fact_sheet(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested fact-sheet dictionaries into dotted paths.

    Lists are retained as leaf values so a field such as ``exclusions`` can be
    cited even though it is not a scalar.
    """
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_fact_sheet(value, path))
        else:
            flat[path] = value
    return flat


def _values_for(catalog: dict[str, Any], paths: list[str]) -> list[Any]:
    values: list[Any] = []
    for path in paths:
        if path not in catalog:
            raise LLMValidationError(f"cited field does not exist: {path!r}")
        values.append(catalog[path])
    return values


def _guard(text: str, values: list[Any]) -> None:
    allowed: set[str] = set()
    for value in values:
        allowed |= _allowed_tokens(value)
    try:
        guard_numbers(text, allowed)
    except GroundingError as exc:
        raise LLMValidationError(str(exc)) from exc


def _allowed_tokens(value: Any) -> set[str]:
    if isinstance(value, bool):
        return set()
    if isinstance(value, (int, float)):
        return numeric_tokens(float(value))
    if isinstance(value, str):
        tokens: set[str] = set()
        if _ISO_DATE.fullmatch(value):
            tokens.add(value)
        try:
            tokens |= numeric_tokens(float(value))
        except (TypeError, ValueError):
            pass
        return tokens
    if isinstance(value, (list, tuple)):
        tokens: set[str] = set()
        for item in value:
            tokens |= _allowed_tokens(item)
        return tokens
    return set()


def _evidence_for(path: str, value: Any, as_of: date) -> Evidence:
    source = f"LLM citation: {path}"
    if len(source) > 120:
        source = source[:117] + "..."
    claim = f"{path} = {_display(value)}"
    if len(claim) > 400:
        claim = claim[:397] + "..."
    numeric = _as_float(value)
    return Evidence(source=source, as_of=as_of, claim=claim, value=numeric)


def _display(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return f"{float(value):.6g}"
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str)


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


__all__ = [
    "LLMReportPayload",
    "LLMSignalPayload",
    "LLMValidationError",
    "build_prompt",
    "flatten_fact_sheet",
    "validate_llm_report",
]
