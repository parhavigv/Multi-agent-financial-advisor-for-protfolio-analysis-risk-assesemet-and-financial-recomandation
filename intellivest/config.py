"""Environment-driven configuration.

All settings have safe defaults so the system runs offline and reproducibly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else value.strip()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    path = Path(raw.strip())
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


@dataclass(frozen=True)
class Settings:
    """Immutable runtime configuration."""

    seed: int = 20261023

    # Data window
    data_start: str = "2018-01-01"
    data_end: str = "2026-06-30"
    lookback_days: int = 756
    risk_free_rate: float = 0.065  # fallback only; prefer provider.get_risk_free()
    trading_days: int = 252

    # Portfolio construction
    max_weight: float = 0.30
    max_sector_weight: float = 0.45
    turnover_penalty: float = 0.0020
    transaction_cost_bps: float = 12.0
    max_repair_iterations: int = 4

    # Black-Litterman view blending
    bl_tau: float = 0.05
    bl_view_shift: float = 0.03
    bl_max_total_influence: float = 0.12

    # LLM layer
    llm_provider: str = "deterministic"
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str | None = None
    llm_timeout_s: float = 30.0

    # Storage
    results_dir: Path = PROJECT_ROOT / "results"
    audits_dir: Path = PROJECT_ROOT / "audits"
    data_dir: Path = PROJECT_ROOT / "data"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            seed=_env_int("INTELLIVEST_SEED", 20261023),
            data_start=_env_str("INTELLIVEST_DATA_START", "2018-01-01"),
            data_end=_env_str("INTELLIVEST_DATA_END", "2026-06-30"),
            lookback_days=_env_int("INTELLIVEST_LOOKBACK_DAYS", 756),
            risk_free_rate=_env_float("INTELLIVEST_RISK_FREE_RATE", 0.065),
            trading_days=_env_int("INTELLIVEST_TRADING_DAYS", 252),
            max_weight=_env_float("INTELLIVEST_MAX_WEIGHT", 0.30),
            max_sector_weight=_env_float("INTELLIVEST_MAX_SECTOR_WEIGHT", 0.45),
            turnover_penalty=_env_float("INTELLIVEST_TURNOVER_PENALTY", 0.0020),
            transaction_cost_bps=_env_float("INTELLIVEST_TRANSACTION_COST_BPS", 12.0),
            max_repair_iterations=_env_int("INTELLIVEST_MAX_REPAIR_ITERATIONS", 4),
            bl_tau=_env_float("INTELLIVEST_BL_TAU", 0.05),
            bl_view_shift=_env_float("INTELLIVEST_BL_VIEW_SHIFT", 0.03),
            bl_max_total_influence=_env_float("INTELLIVEST_BL_MAX_TOTAL_INFLUENCE", 0.12),
            llm_provider=_env_str("INTELLIVEST_LLM_PROVIDER", "deterministic").lower(),
            llm_model=_env_str("INTELLIVEST_LLM_MODEL", "gpt-4o-mini"),
            llm_base_url=_env_str("INTELLIVEST_LLM_BASE_URL", "https://api.openai.com/v1"),
            llm_api_key=os.getenv("INTELLIVEST_LLM_API_KEY") or None,
            llm_timeout_s=_env_float("INTELLIVEST_LLM_TIMEOUT_S", 30.0),
            results_dir=_env_path("INTELLIVEST_RESULTS_DIR", PROJECT_ROOT / "results"),
            audits_dir=_env_path("INTELLIVEST_AUDITS_DIR", PROJECT_ROOT / "audits"),
            data_dir=_env_path("INTELLIVEST_DATA_DIR", PROJECT_ROOT / "data"),
        )

    def ensure_directories(self) -> None:
        for path in (self.results_dir, self.audits_dir, self.data_dir):
            path.mkdir(parents=True, exist_ok=True)

    @property
    def llm_enabled(self) -> bool:
        return self.llm_provider != "deterministic" and bool(self.llm_api_key)


def load_settings() -> Settings:
    """Load settings from the environment (no caching, so tests can override)."""
    settings = Settings.from_env()
    settings.ensure_directories()
    return settings
