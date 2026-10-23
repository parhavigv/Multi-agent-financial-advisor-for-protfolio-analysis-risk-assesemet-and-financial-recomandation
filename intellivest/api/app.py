"""FastAPI service exposing the advisory pipeline and its audit trail."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from intellivest import __version__
from intellivest.config import Settings, load_settings
from intellivest.contracts import AdviceResponse, DecisionTrace, InvestorProfile
from intellivest.data.universe import describe_universe
from intellivest.orchestration.orchestrator import AdvisoryEngine

_EXAMPLE_PROFILE = {
    "investor_id": "DEMO-001",
    "as_of": "2026-06-30",
    "age": 34,
    "horizon_years": 12,
    "risk_tolerance": 4,
    "loss_tolerance_pct": 0.20,
    "goal": "growth",
    "liquidity_need": "low",
    "current_holdings": {"SPX": 0.35, "TECH": 0.30, "GOLD": 0.10, "GOVT": 0.25},
    "exclusions": [],
    "notes": "Reference investor used in the evaluation harness.",
}


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    provider: str
    universe_size: int
    llm_provider: str
    llm_enabled: bool


class TraceSummary(BaseModel):
    trace_id: str
    investor_id: str
    as_of: date
    hash: str
    previous_hash: str
    suitability_passed: bool
    steps: int


class AuditVerification(BaseModel):
    ok: bool
    message: str


class AdviseRequest(InvestorProfile):
    """The request body *is* the validated investor profile."""


def create_app(settings: Settings | None = None, engine: AdvisoryEngine | None = None) -> FastAPI:
    resolved = settings or load_settings()
    advisory = engine or AdvisoryEngine(resolved)

    app = FastAPI(
        title="INTELLIVEST-MAS",
        version=__version__,
        description=(
            "Hybrid multi-agent financial advisory: portfolio analysis, risk assessment and "
            "suitability-aware investment recommendation with a hash-chained decision trace."
        ),
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            version=__version__,
            provider=type(advisory.provider).__name__,
            universe_size=len(describe_universe()),
            llm_provider=resolved.llm_provider,
            llm_enabled=resolved.llm_enabled,
        )

    @app.get("/universe")
    def universe() -> list[dict[str, Any]]:
        return describe_universe()

    @app.get("/profiles/example", response_model=InvestorProfile)
    def example_profile() -> InvestorProfile:
        return InvestorProfile(**_EXAMPLE_PROFILE)

    @app.post("/advise", response_model=AdviceResponse)
    def advise(request: AdviseRequest) -> AdviceResponse:
        profile = InvestorProfile(**request.model_dump())
        try:
            trace = advisory.advise(profile)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return AdviceResponse(
            trace_id=trace.trace_id,
            recommendation=trace.recommendation,
            agents=trace.agents,
            explanation=trace.explanation,
            steps=trace.steps,
        )

    @app.get("/trace/{trace_id}", response_model=DecisionTrace)
    def trace(trace_id: str) -> DecisionTrace:
        path: Path = resolved.audits_dir / "traces" / f"{trace_id}.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail="trace not found")
        return DecisionTrace.model_validate(json.loads(path.read_text(encoding="utf-8")))

    @app.get("/audit/verify", response_model=AuditVerification)
    def verify() -> AuditVerification:
        ok, message = advisory.verify_audit_chain()
        return AuditVerification(ok=ok, message=message)

    return app


app = create_app()
