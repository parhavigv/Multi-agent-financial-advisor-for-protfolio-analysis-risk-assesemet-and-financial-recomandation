"""Time-based splits as per evaluation protocol.

dev: up to 2021-12-31
validation: 2022-2023
test: 2024 onwards (touched once at the end)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class TimeSplit:
    dev_end: date = date(2021, 12, 31)
    val_start: date = date(2022, 1, 1)
    val_end: date = date(2023, 12, 31)
    test_start: date = date(2024, 1, 1)

    def split(self, as_of: date) -> str:
        if as_of <= self.dev_end:
            return "dev"
        if self.val_start <= as_of <= self.val_end:
            return "validation"
        if as_of >= self.test_start:
            return "test"
        return "dev"


DEFAULT_SPLIT = TimeSplit()
