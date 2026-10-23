"""Decision tracing and tamper-evident audit storage.

Every pipeline stage records a digest of its inputs and outputs. The completed
trace is hashed and chained to the previous trace, so removing or editing an
earlier recommendation breaks the chain. This is the concrete implementation of
the survey's "end-to-end decision traceability" gap.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from intellivest.contracts import TraceStep


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def digest(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


class TraceRecorder:
    def __init__(self) -> None:
        self.steps: list[TraceStep] = []

    @contextmanager
    def step(self, name: str) -> Iterator[dict[str, Any]]:
        """Record a step. Attach ``inputs`` / ``outputs`` / notes to the dict."""
        started = datetime.now(timezone.utc)
        bucket: dict[str, Any] = {"inputs": None, "outputs": None, "notes": None, "status": "ok"}
        try:
            yield bucket
        except Exception:
            bucket["status"] = "failed"
            raise
        finally:
            finished = datetime.now(timezone.utc)
            self.steps.append(
                TraceStep(
                    name=name,
                    started_at=started,
                    duration_ms=round((finished - started).total_seconds() * 1000.0, 3),
                    status=bucket.get("status", "ok"),
                    inputs_digest=digest(bucket.get("inputs")) if bucket.get("inputs") is not None else "",
                    output_digest=digest(bucket.get("outputs")) if bucket.get("outputs") is not None else "",
                    notes=bucket.get("notes"),
                )
            )

    def mark_degraded(self) -> None:
        if self.steps:
            self.steps[-1] = self.steps[-1].model_copy(update={"status": "degraded"})


class AuditChain:
    """Append-only JSONL chain plus per-trace snapshots."""

    def __init__(self, audits_dir: Path) -> None:
        self.audits_dir = audits_dir
        self.chain_path = audits_dir / "chain.jsonl"
        self.traces_dir = audits_dir / "traces"
        self.traces_dir.mkdir(parents=True, exist_ok=True)

    def last_hash(self) -> str:
        if not self.chain_path.exists():
            return "GENESIS"
        last = "GENESIS"
        with self.chain_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    last = json.loads(line).get("hash", last)
                except json.JSONDecodeError:
                    continue
        return last

    def seal(self, payload: dict[str, Any]) -> tuple[str, str]:
        previous = self.last_hash()
        body = dict(payload)
        body["previous_hash"] = previous
        final_hash = digest(body)
        body["hash"] = final_hash
        record = {
            "trace_id": payload.get("trace_id"),
            "investor_id": payload.get("investor_id"),
            "created_at": payload.get("created_at"),
            "previous_hash": previous,
            "hash": final_hash,
        }
        with self.chain_path.open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(record) + "\n")
        if payload.get("trace_id"):
            (self.traces_dir / f"{payload['trace_id']}.json").write_text(
                json.dumps(body, indent=2, default=str), encoding="utf-8"
            )
        return previous, final_hash

    def verify(self) -> tuple[bool, str]:
        """Recompute the chain and every trace body. Returns (ok, message).

        Two independent checks run per record: the ``previous_hash`` link, and
        the content digest recomputed from the stored trace snapshot. Editing a
        recommendation therefore invalidates its hash even when the record line
        itself is left untouched.
        """
        if not self.chain_path.exists():
            return True, "empty chain"
        expected_previous = "GENESIS"
        count = 0
        with self.chain_path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                count += 1
                record = json.loads(line)
                if record.get("previous_hash") != expected_previous:
                    return False, f"line {index}: previous_hash mismatch"
                trace_id = record.get("trace_id")
                trace_path = self.traces_dir / f"{trace_id}.json"
                if not trace_id or not trace_path.exists():
                    return False, f"line {index}: trace snapshot missing"
                sealed = json.loads(trace_path.read_text(encoding="utf-8"))
                claimed = sealed.get("hash")
                recomputed = digest({key: value for key, value in sealed.items() if key != "hash"})
                if claimed != record.get("hash") or recomputed != claimed:
                    return False, f"line {index}: content hash mismatch"
                expected_previous = record.get("hash")
        return True, f"verified {count} records"


__all__ = ["AuditChain", "TraceRecorder", "canonical_json", "digest"]
