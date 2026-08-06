#!/usr/bin/env python3
"""Standalone task-queue runner -- NO Claude Code / LLM dependency at runtime.

Built 2026-08-04 (resource-management infrastructure, .claude/plans/snazzy-cuddling-creek.md
Part 0.4) so pre-defined work can keep progressing via a plain, unattended process even when
Claude Code itself is unavailable (e.g. weekly usage limit exhausted until reset).

Design, in one paragraph: the queue file (JSON) IS the state -- one list of task objects, each
with a `status` that only ever moves pending -> running -> (done | failed), written atomically
(temp file + os.replace, POSIX-atomic) after every single transition, not just at the end. On
startup, before running anything, the runner scans for any task already marked "running" -- that
means a previous run was interrupted mid-task -- marks it "interrupted" and HALTS immediately,
never auto-resuming or re-running that task's own command blindly (no LLM available standalone to
judge whether that's safe). Only genuinely "pending" tasks run, strictly in order; the first
non-"done" status the runner encounters (interrupted, failed, or an already-"running" task from a
prior crash) stops the whole queue for manual review, it never skips ahead to later pending tasks.

Usage:
  # write a trivial test queue (verifying the mechanics, not real work)
  python task_queue_runner.py --queue .tmp/task_queue/queue.json --init-test-queue

  # run the queue (repeatable: safe to run again after a completed run, after a resolved
  # failure/interruption, or after being killed mid-task)
  python task_queue_runner.py --queue .tmp/task_queue/queue.json

Queue file format: a JSON list of task objects:
  {"id": "name", "cmd": ["argv0", "arg1", ...], "gate": {"type": "exit_zero"}, "status": "pending"}
Gate types (kept deliberately minimal, not a general expression engine):
  - exit_zero (default): pass iff the subprocess returned exit code 0.
  - log_regex_match: {"type": "log_regex_match", "pattern": "..."} -- pass iff the pattern is
    found anywhere in the task's combined stdout+stderr.
  - log_regex_absent: {"type": "log_regex_absent", "pattern": "..."} -- pass iff the pattern is
    NOT found (e.g. a "NaN" or "Traceback" absence check).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_QUEUE_PATH = Path(__file__).resolve().parent.parent.parent / ".tmp/task_queue/queue.json"
VALID_STATUSES = {"pending", "running", "done", "failed", "interrupted"}


def load_queue(path: Path) -> list[dict]:
    with open(path) as f:
        queue = json.load(f)
    for task in queue:
        if task.get("status") not in VALID_STATUSES:
            raise ValueError(f"task {task.get('id')!r} has invalid status {task.get('status')!r}")
    return queue


def save_queue_atomic(path: Path, queue: list[dict]) -> None:
    """Write to a temp file in the SAME directory (so os.replace stays on one filesystem, a
    requirement for the rename to be atomic), then atomically replace the real path -- a kill at
    any point leaves either the old complete file or the new complete file, never a half-written
    one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=".queue_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(queue, f, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def check_gate(gate: dict, returncode: int, output: str) -> bool:
    gtype = gate.get("type", "exit_zero")
    if gtype == "exit_zero":
        return returncode == 0
    if gtype == "log_regex_match":
        return re.search(gate["pattern"], output) is not None
    if gtype == "log_regex_absent":
        return re.search(gate["pattern"], output) is None
    raise ValueError(f"unknown gate type: {gtype!r}")


def run_queue(path: Path) -> int:
    queue = load_queue(path)
    n = len(queue)

    # Interrupted-task check FIRST, before running anything new -- this is the core safety
    # requirement, checked before any new work starts.
    for i, task in enumerate(queue):
        if task["status"] == "running":
            task["status"] = "interrupted"
            save_queue_atomic(path, queue)
            print(f"[queue] {i + 1}/{n} '{task['id']}': found RUNNING from a previous "
                  f"interrupted run -- marked INTERRUPTED, halting. Not auto-resumed: review "
                  f"the task and its last output, then edit the queue file's status to "
                  f"'pending' (or 'done', if it actually finished) before re-running.")
            return 1

    for i, task in enumerate(queue):
        status = task["status"]
        if status == "done":
            continue
        if status in ("failed", "interrupted"):
            print(f"[queue] {i + 1}/{n} '{task['id']}': status={status}, halting -- resolve "
                  f"manually (edit the queue file) before continuing.")
            return 1

        print(f"[queue] {i + 1}/{n} '{task['id']}': running...")
        task["status"] = "running"
        save_queue_atomic(path, queue)

        try:
            proc = subprocess.run(
                task["cmd"], capture_output=True, text=True,
                timeout=task.get("timeout_sec"),
            )
            output = proc.stdout + proc.stderr
            returncode = proc.returncode
        except subprocess.TimeoutExpired as e:
            output = (e.stdout or "") + (e.stderr or "")
            returncode = -1

        gate = task.get("gate", {"type": "exit_zero"})
        passed = check_gate(gate, returncode, output)
        task["status"] = "done" if passed else "failed"
        task["returncode"] = returncode
        task["last_output_tail"] = output[-2000:]
        save_queue_atomic(path, queue)

        result = "PASS" if passed else f"FAIL (gate: {gate.get('type')})"
        print(f"[queue] {i + 1}/{n} '{task['id']}': {result}")
        if not passed:
            return 1

    print(f"[queue] all {n} tasks done.")
    return 0


TEST_QUEUE = [
    {"id": "quick_echo", "cmd": ["echo", "hello"],
     "gate": {"type": "log_regex_match", "pattern": "hello"}, "status": "pending"},
    {"id": "slow_sleep", "cmd": ["sleep", "8"],
     "gate": {"type": "exit_zero"}, "status": "pending"},
    {"id": "final_echo", "cmd": ["echo", "final"],
     "gate": {"type": "exit_zero"}, "status": "pending"},
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue", type=Path, default=DEFAULT_QUEUE_PATH)
    ap.add_argument("--init-test-queue", action="store_true",
                     help="Write a trivial placeholder queue to --queue, for verifying the "
                          "runner's own mechanics -- not real work.")
    args = ap.parse_args()

    if args.init_test_queue:
        save_queue_atomic(args.queue, [dict(t) for t in TEST_QUEUE])
        print(f"wrote trivial test queue ({len(TEST_QUEUE)} tasks) to {args.queue}")
        return 0

    if not args.queue.exists():
        print(f"no queue file at {args.queue} -- run with --init-test-queue first, or create one.")
        return 1

    return run_queue(args.queue)


if __name__ == "__main__":
    sys.exit(main())
