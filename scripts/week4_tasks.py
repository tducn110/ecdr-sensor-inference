#!/usr/bin/env python3
"""scripts/week4_tasks.py — Task claim, lock management, receipt verification, and review CLI.

Implements the official Week 4 coordination protocol defined in ORCHESTRATOR.md:
- Ready task calculation with dependency and write-path collision detection
- Atomic file locking (fcntl.flock) and registry updates
- Path normalization to prevent lock aliasing
- Stale lock protection (timeout alone cannot steal tasks)
- Receipt schema and completeness validation
- Separation of implementer action from independent reviewer action
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, List, Optional, Set, Tuple

DEFAULT_REGISTRY = Path(__file__).resolve().parents[1] / "04_SYNTHESIS/week4/tasks.json"
REPO_ROOT = Path(__file__).resolve().parents[1]


class TaskError(Exception):
    """Domain error during task coordination operations."""


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_path(p: str, root: Path = REPO_ROOT) -> str:
    """Normalize file path relative to repo root to prevent lock aliasing."""
    path_obj = Path(p)
    if path_obj.is_absolute():
        try:
            rel = path_obj.relative_to(root)
            return os.path.normpath(str(rel))
        except ValueError:
            return os.path.normpath(str(path_obj))
    return os.path.normpath(str(path_obj))


class RegistryManager:
    """Atomic registry manager with OS-level advisory locking."""

    def __init__(self, registry_path: Path):
        self.registry_path = registry_path.resolve()
        self.lock_path = self.registry_path.with_suffix(".lock")

    def _load_data_unlocked(self) -> Dict[str, Any]:
        if not self.registry_path.is_file():
            raise TaskError(f"Registry not found at {self.registry_path}")
        with open(self.registry_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_data_atomic(self, data: Dict[str, Any]) -> None:
        """Atomic write using temporary file and atomic rename."""
        parent = self.registry_path.parent
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=parent, delete=False, encoding="utf-8") as tf:
            json.dump(data, tf, indent=2)
            tf.flush()
            os.fsync(tf.fileno())
            temp_name = tf.name
        os.replace(temp_name, self.registry_path)

    def with_lock(self, func, *args, **kwargs) -> Any:
        """Execute func holding exclusive file lock on the registry."""
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.lock_path, "w") as lock_file:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                data = self._load_data_unlocked()
                result, modified = func(data, *args, **kwargs)
                if modified:
                    self._save_data_atomic(data)
                return result
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


# ==============================================================================
# Protocol Operations
# ==============================================================================

def get_ready_tasks(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return all TODO tasks whose dependencies are satisfied and paths unconflicted."""
    tasks = data.get("tasks", [])
    task_map = {t["id"]: t for t in tasks}

    # Collect all currently active locked paths
    active_locked_paths: Set[str] = set()
    for lock in data.get("locks", []):
        for p in lock.get("paths", []):
            active_locked_paths.add(_normalize_path(p))

    ready = []
    for t in tasks:
        if t.get("status") != "TODO":
            continue

        # Check dependencies
        deps_satisfied = True
        for dep_id in t.get("dependencies", []):
            dep_task = task_map.get(dep_id)
            if not dep_task:
                deps_satisfied = False
                break
            dep_status = dep_task.get("status")
            if dep_status == "VERIFIED":
                continue
            elif dep_status == "VERIFIED_LIMITED":
                # Only ready if downstream explicitly documents handling
                if not t.get("dependency_rule"):
                    deps_satisfied = False
                    break
            else:
                deps_satisfied = False
                break

        if not deps_satisfied:
            continue

        # Check write set collision against currently locked paths
        task_paths = {_normalize_path(p) for p in t.get("allowed_write_paths", [])}
        if task_paths.intersection(active_locked_paths):
            continue

        ready.append({
            "id": t["id"],
            "title": t["title"],
            "milestone": t["milestone"],
            "dependencies": t.get("dependencies", []),
            "allowed_write_paths": t.get("allowed_write_paths", []),
        })

    return ready


def claim_task(
    data: Dict[str, Any],
    task_id: str,
    agent_id: str,
    base_commit: str = "",
) -> Tuple[Dict[str, Any], bool]:
    """Claim a task, acquiring exclusive path locks atomically."""
    task_map = {t["id"]: t for t in data.get("tasks", [])}
    target = task_map.get(task_id)
    if not target:
        raise TaskError(f"Task {task_id} not found in registry")

    if target.get("status") != "TODO":
        raise TaskError(
            f"Cannot claim task {task_id}: status is {target.get('status')} (must be TODO)"
        )

    # Check dependencies
    for dep_id in target.get("dependencies", []):
        dep_task = task_map.get(dep_id)
        if not dep_task or dep_task.get("status") not in ("VERIFIED", "VERIFIED_LIMITED"):
            dep_status = dep_task.get("status") if dep_task else "MISSING"
            raise TaskError(
                f"Dependency {dep_id} is not satisfied (status: {dep_status})"
            )

    # Check write set collision
    normalized_paths = [_normalize_path(p) for p in target.get("allowed_write_paths", [])]
    existing_locks = data.get("locks", [])
    for lock in existing_locks:
        lock_paths = {_normalize_path(p) for p in lock.get("paths", [])}
        overlap = set(normalized_paths).intersection(lock_paths)
        if overlap:
            raise TaskError(
                f"Write path collision on {sorted(overlap)} with task {lock.get('task_id')}"
            )

    # Acquire lock and set claim
    now = _now_utc()
    existing_locks.append({
        "task_id": task_id,
        "paths": normalized_paths,
        "locked_at_utc": now,
        "agent_id": agent_id,
    })
    data["locks"] = existing_locks

    target["status"] = "IN_PROGRESS"
    target["assignee"] = agent_id
    target["claim"] = {
        "claimed_at_utc": now,
        "agent_id": agent_id,
        "base_commit": base_commit,
    }

    return target, True


def validate_receipt_file(receipt_path: Path, expected_task_id: str) -> Dict[str, Any]:
    """Validate task receipt structure, completeness, and evidence integrity."""
    if not receipt_path.is_file():
        raise TaskError(f"Receipt file does not exist: {receipt_path}")

    try:
        data = json.loads(receipt_path.read_text(encoding="utf-8"))
    except Exception as err:
        raise TaskError(f"Receipt is not valid JSON: {err}") from err

    if data.get("template_only") is True:
        raise TaskError("Receipt has template_only=True; real execution receipt required")

    if data.get("task_id") != expected_task_id:
        raise TaskError(
            f"Receipt task_id '{data.get('task_id')}' mismatch expected '{expected_task_id}'"
        )

    if not data.get("commands") or len(data["commands"]) == 0:
        raise TaskError("Receipt contains no executed command records")

    for cmd in data["commands"]:
        if "exit_code" not in cmd:
            raise TaskError(f"Command record missing exit_code: {cmd}")
        if cmd["exit_code"] != 0:
            raise TaskError(f"Command failed with exit_code {cmd['exit_code']}: {cmd.get('command')}")

    if not data.get("requirement_evidence") or len(data["requirement_evidence"]) == 0:
        raise TaskError("Receipt contains no requirement evidence mappings")

    for req in data["requirement_evidence"]:
        if req.get("status") not in ("PASS", "VERIFIED"):
            raise TaskError(f"Requirement {req.get('requirement_id')} has non-pass status: {req.get('status')}")

    return data


def submit_task(
    data: Dict[str, Any],
    task_id: str,
    receipt_path_str: str,
    agent_id: Optional[str] = None,
) -> Tuple[Dict[str, Any], bool]:
    """Submit a task for review with a verified receipt."""
    task_map = {t["id"]: t for t in data.get("tasks", [])}
    target = task_map.get(task_id)
    if not target:
        raise TaskError(f"Task {task_id} not found in registry")

    if target.get("status") not in ("IN_PROGRESS", "REWORK"):
        raise TaskError(
            f"Cannot submit task {task_id}: current status is {target.get('status')}"
        )

    if agent_id and target.get("assignee") and target.get("assignee") != agent_id:
        raise TaskError(
            f"Submitter agent '{agent_id}' does not match assigned agent '{target.get('assignee')}'"
        )

    # Validate receipt file
    receipt_path = Path(receipt_path_str)
    receipt_data = validate_receipt_file(receipt_path, task_id)

    target["status"] = "REVIEW"
    target["execution_receipt"] = str(receipt_path)
    target["submitted_at_utc"] = _now_utc()

    return target, True


def review_task(
    data: Dict[str, Any],
    task_id: str,
    reviewer_id: str,
    decision: str,
    notes: str = "",
    allow_self_review: bool = False,
) -> Tuple[Dict[str, Any], bool]:
    """Record independent review decision and release locks upon verification."""
    task_map = {t["id"]: t for t in data.get("tasks", [])}
    target = task_map.get(task_id)
    if not target:
        raise TaskError(f"Task {task_id} not found in registry")

    if target.get("status") != "REVIEW":
        raise TaskError(
            f"Cannot review task {task_id}: current status is {target.get('status')} (must be REVIEW)"
        )

    if not allow_self_review and target.get("assignee") == reviewer_id:
        raise TaskError(
            f"Reviewer '{reviewer_id}' cannot review their own task. Independent reviewer required."
        )

    valid_decisions = ("VERIFIED", "VERIFIED_LIMITED", "REWORK")
    if decision not in valid_decisions:
        raise TaskError(f"Invalid review decision '{decision}', must be one of {valid_decisions}")

    now = _now_utc()
    target["review"] = {
        "reviewer_id": reviewer_id,
        "decision": decision,
        "reviewed_at_utc": now,
        "notes": notes,
    }

    if decision in ("VERIFIED", "VERIFIED_LIMITED"):
        target["status"] = decision
        # Release locks upon successful verification
        data["locks"] = [l for l in data.get("locks", []) if l.get("task_id") != task_id]
    else:  # REWORK
        target["status"] = "REWORK"
        # Keep locks for rework by assignee

    return target, True


def release_task(
    data: Dict[str, Any],
    task_id: str,
    reason: str,
) -> Tuple[Dict[str, Any], bool]:
    """Explicitly release lock and reset claim with mandatory justification."""
    if not reason or not reason.strip():
        raise TaskError("Cannot release task without a non-empty reason")

    task_map = {t["id"]: t for t in data.get("tasks", [])}
    target = task_map.get(task_id)
    if not target:
        raise TaskError(f"Task {task_id} not found in registry")

    # Release locks
    initial_lock_count = len(data.get("locks", []))
    data["locks"] = [l for l in data.get("locks", []) if l.get("task_id") != task_id]
    locks_released = initial_lock_count - len(data["locks"])

    if target.get("status") in ("IN_PROGRESS", "REWORK"):
        target["status"] = "TODO"
        target["assignee"] = None
        target["claim"] = None
        target["release_history"] = target.get("release_history", []) + [{
            "released_at_utc": _now_utc(),
            "reason": reason,
            "locks_released": locks_released,
        }]

    return target, True


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Week 4 task coordination, atomic claims, and receipt review CLI."
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY,
        help="Path to tasks.json registry file",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ready
    subparsers.add_parser("ready", help="List tasks currently ready to claim")

    # claim
    claim_p = subparsers.add_parser("claim", help="Claim a ready task with exclusive lock")
    claim_p.add_argument("task_id", help="Task ID (e.g. W4-004)")
    claim_p.add_argument("--agent", required=True, help="Agent/Assignee ID")
    claim_p.add_argument("--commit", default="", help="Base commit hash")

    # submit
    submit_p = subparsers.add_parser("submit", help="Submit task with verified execution receipt")
    submit_p.add_argument("task_id", help="Task ID")
    submit_p.add_argument("--receipt", required=True, help="Path to receipt.json")
    submit_p.add_argument("--agent", help="Submitting agent ID (optional verification)")

    # review
    review_p = subparsers.add_parser("review", help="Submit independent review decision")
    review_p.add_argument("task_id", help="Task ID")
    review_p.add_argument("--reviewer", required=True, help="Independent reviewer ID")
    review_p.add_argument(
        "--decision",
        required=True,
        choices=["VERIFIED", "VERIFIED_LIMITED", "REWORK"],
        help="Review decision",
    )
    review_p.add_argument("--notes", default="", help="Review notes or feedback")
    review_p.add_argument("--allow-self-review", action="store_true", help="Allow author self-review")

    # release
    release_p = subparsers.add_parser("release", help="Explicitly release lock and reset claim")
    release_p.add_argument("task_id", help="Task ID")
    release_p.add_argument("--reason", required=True, help="Mandatory release justification")

    # status
    status_p = subparsers.add_parser("status", help="Show registry status or task details")
    status_p.add_argument("task_id", nargs="?", help="Optional specific Task ID")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    mgr = RegistryManager(args.registry)

    try:
        if args.command == "ready":
            tasks = mgr.with_lock(lambda d: (get_ready_tasks(d), False))
            print(f"Ready tasks ({len(tasks)}):")
            for t in tasks:
                paths_str = ", ".join(t["allowed_write_paths"])
                print(f"  [{t['id']}] {t['title']} (paths: {paths_str})")
            return 0

        elif args.command == "claim":
            target = mgr.with_lock(claim_task, args.task_id, args.agent, args.commit)
            print(f"Successfully claimed {args.task_id} for agent '{args.agent}'.")
            return 0

        elif args.command == "submit":
            target = mgr.with_lock(submit_task, args.task_id, args.receipt, args.agent)
            print(f"Successfully submitted {args.task_id} with receipt {args.receipt}.")
            return 0

        elif args.command == "review":
            target = mgr.with_lock(
                review_task,
                args.task_id,
                args.reviewer,
                args.decision,
                args.notes,
                args.allow_self_review,
            )
            print(f"Successfully reviewed {args.task_id}: decision = {args.decision}.")
            return 0

        elif args.command == "release":
            target = mgr.with_lock(release_task, args.task_id, args.reason)
            print(f"Successfully released {args.task_id} (reason: {args.reason}).")
            return 0

        elif args.command == "status":
            def _status_action(data: Dict[str, Any]):
                if args.task_id:
                    task_map = {t["id"]: t for t in data.get("tasks", [])}
                    t = task_map.get(args.task_id)
                    if not t:
                        raise TaskError(f"Task {args.task_id} not found")
                    print(json.dumps(t, indent=2))
                else:
                    counts: Dict[str, int] = {}
                    for t in data.get("tasks", []):
                        st = t.get("status", "UNKNOWN")
                        counts[st] = counts.get(st, 0) + 1
                    print(f"Tasks overview: {counts}")
                    print(f"Active locks ({len(data.get('locks', []))}):")
                    for l in data.get("locks", []):
                        print(f"  Task {l['task_id']} by {l['agent_id']}: {l['paths']}")
                return None, False

            mgr.with_lock(_status_action)
            return 0

    except TaskError as err:
        print(f"TaskError: {err}", file=sys.stderr)
        return 1
    except Exception as err:
        print(f"Unexpected error: {err}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
