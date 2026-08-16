"""Gen9 v2 hard-lock mechanism (decisions.md 2026-08-14).

The user's hierarchy has three write-once lock tiers:
  frame_lock  -- set at step 26 (frames locked forever)
  sfx_lock    -- set at step 44 (frames + SFX locked forever)
  spiky_lock  -- set at step 54 (pipeline complete)

Literal semantics, enforced by construction:
- delete()/restore() are the ONLY ways to change the mask. Both clip the
  incoming pixel set against every set lock. With strict=True (default)
  any clipped pixel RAISES LockViolation -- a stage that should never
  touch locked area cannot silently do so. Rectangle-scoped spiky ops
  pass strict=False: there the clipping IS the frame protection the
  etalon shows (the spiky ellipse portion over the locked pale panel is
  untouched), and the clipped count is recorded for the report.
- Each lock_*() snapshots the mask; verify_locks() re-checks every
  snapshot bit-for-bit. The staged runner calls verify_locks() after
  every stage, so a violation is caught at the stage that caused it.
- pending: regions of the background zone whose fate is not yet decided
  (SFX candidates before 44, spiky regions before 54). lock_sfx() /
  lock_spiky() RAISE if any pending region of their tier is unresolved
  -- "every region explicitly resolved, none defaulted" is a hard
  invariant, not a convention.
"""
from __future__ import annotations

import numpy as np


class LockViolation(RuntimeError):
    pass


class PendingUnresolved(RuntimeError):
    pass


class PageState:
    TIERS = ('frame', 'sfx', 'spiky')

    def __init__(self, shape: tuple[int, int]):
        self.mask = np.zeros(shape, bool)          # True = deleted
        self.locks: dict[str, np.ndarray | None] = {t: None for t in self.TIERS}
        self._snaps: dict[str, np.ndarray] = {}
        self.pending: dict[str, dict] = {}         # id -> {tier, mask/bbox, status}
        self.clipped_log: list[tuple[str, int]] = []

    # -- mask edits ------------------------------------------------------
    def _apply(self, px: np.ndarray, value: bool, strict: bool, tag: str):
        px = px.astype(bool, copy=True)
        clipped = 0
        for t, lk in self.locks.items():
            if lk is not None:
                hit = px & lk
                n = int(hit.sum())
                if n:
                    if strict:
                        raise LockViolation(
                            f'{tag}: {n} px write into {t}_lock (strict)')
                    px &= ~lk
                    clipped += n
        if clipped:
            self.clipped_log.append((tag, clipped))
        self.mask[px] = value

    def delete(self, px: np.ndarray, strict: bool = True, tag: str = 'delete'):
        self._apply(px, True, strict, tag)

    def restore(self, px: np.ndarray, strict: bool = True, tag: str = 'restore'):
        self._apply(px, False, strict, tag)

    # -- pending registry ------------------------------------------------
    def add_pending(self, rid: str, tier: str, **info):
        assert tier in self.TIERS
        self.pending[rid] = dict(tier=tier, status='unresolved', **info)

    def resolve(self, rid: str, outcome: str):
        """outcome: 'keep' | 'delete' | 'restore-fringe' | ... (recorded)."""
        self.pending[rid]['status'] = outcome

    def unresolved(self, tier: str) -> list[str]:
        return [r for r, v in self.pending.items()
                if v['tier'] == tier and v['status'] == 'unresolved']

    # -- locks -----------------------------------------------------------
    def _lock(self, tier: str, region: np.ndarray):
        if self.locks[tier] is not None:
            raise LockViolation(f'{tier}_lock already set (write-once)')
        un = self.unresolved(tier)
        if un:
            raise PendingUnresolved(f'{tier}_lock with unresolved: {un}')
        self.locks[tier] = region.astype(bool, copy=True)
        self._snaps[tier] = self.mask[self.locks[tier]].copy()

    def lock_frames(self, region: np.ndarray):
        self._lock('frame', region)

    def lock_sfx(self, region: np.ndarray):
        self._lock('sfx', region)

    def lock_spiky(self, region: np.ndarray):
        self._lock('spiky', region)

    def verify_locks(self):
        """Bit-exact: mask under every set lock unchanged since locking."""
        for t, lk in self.locks.items():
            if lk is not None:
                if not np.array_equal(self.mask[lk], self._snaps[t]):
                    raise LockViolation(f'{t}_lock content changed')
        return True
