"""Chronological splits with purge/embargo."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Split:
    train: tuple[int, int]
    val: tuple[int, int]
    test: tuple[int, int]


def purged_time_split(n: int, purge: int = 5, embargo: int = 5):
    """Return (train, val, test) index ranges with purge/embargo."""
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    # apply purge/embargo
    train = (0, max(0, train_end - purge))
    val = (train_end + purge, max(train_end + purge, val_end - embargo))
    test = (val_end + embargo, n)
    if val[0] >= val[1]:
        val = (train_end, val_end)
    if test[0] >= test[1]:
        test = (val_end, n)
    if train[0] >= train[1]:
        train = (0, train_end)
    return Split(train=train, val=val, test=test)
