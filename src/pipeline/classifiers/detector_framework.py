"""Detector framework: ONE detection loop, pluggable geometry profiles.

Gen-8 phase 4 (2026-08-10). A profile bundles (a) a candidate generator (the
shape/structural search), (b) independent geometric SIGNALS, each a hard gate with its
own threshold, combined by AND-voting -- the structure the v23 spiky-cloud cascade
validated (13/13 TP, 0/85 FP on its reference set): multiple independent geometric
signals, precision-first, each individually cheap and interpretable.

Classical rules/geometry only (gen-8 standing principle -- no ML anywhere, including
fallbacks). Full-page context: candidate generation always sees the whole page; signals
may window around their candidate but must document any context they drop.

Profiles live in `src/classifiers/profiles/` -- one module per profile exposing
`PROFILE`. Current registry:
  spiky_cloud   -- ported v23 cascade (equivalence-gated against
                   `pipeline.find_spiky_sites`, which remains the production caller
                   until re-pointing is explicitly decided).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

Region = tuple[int, int, int, int]  # (x0, y0, x1, y1)


@dataclass
class Signal:
    """One independent geometric gate. fn(page_rgb, region) -> measured value;
    accept(value) -> bool. Kept separate so reports can show per-signal values
    (the v20/v23 standard: show the trade-offs, don't hide them in a vote)."""
    name: str
    fn: Callable[[np.ndarray, Region], float]
    accept: Callable[[float], bool]


@dataclass
class Profile:
    """A geometry profile: candidate generator + AND-combined signals."""
    name: str
    candidates: Callable[[np.ndarray], list[Region]]
    signals: list[Signal] = field(default_factory=list)


def detect(page: np.ndarray, profile: Profile,
           explain: bool = False) -> list[Region] | list[dict]:
    """Run a profile: generate candidates on the FULL page, keep those passing every
    signal. With explain=True returns per-candidate dicts (region, per-signal values,
    accepted) instead of bare regions -- for suite/adversarial reporting."""
    out = []
    for region in profile.candidates(page):
        values = {}
        ok = True
        for sig in profile.signals:
            v = sig.fn(page, region)
            values[sig.name] = v
            if not sig.accept(v):
                ok = False
                if not explain:
                    break
        if explain:
            out.append({"region": region, "signals": values, "accepted": ok})
        elif ok:
            out.append(region)
    return out
