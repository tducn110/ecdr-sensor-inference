"""tests/test_week4_task_claims.py — Unit tests for Week 4 task coordination engine.

Verifies:
1. Ready tasks computation with dependency and lock-collision checking.
2. Atomic claim acquisition, exclusive write-path locking, and race collision rejection.
3. Path normalization preventing lock aliasing (e.g. './scripts/a.py' vs 'scripts/a.py').
4. Dependency requirement enforcement (blocking unverified prerequisites).
5. Stale claim release requiring mandatory explicit justification.
6. Execution receipt structural and completeness verification.
7. Independent reviewer enforcement (blocking author self-review).
8. Strict isolation: all tests run on disposable temporary registry fixtures.
"""

import json
from pathlib import Path
import tempfile
import unittest

from scripts.week4_tasks import (
    RegistryManager,
    TaskError,
    claim_task,
    get_ready_tasks,
    release_task,
    review_task,
    submit_task,
    validate_receipt_file,
)


class TestWeek4TaskCoordination(unittest.TestCase):
    """Test suite running exclusively on disposable isolated registry fixtures."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.registry_file = self.temp_path / "tasks.json"

        # Construct minimal test DAG with tasks T1, T2 (depends on T1), T3 (same write path as T1)
        self.initial_data = {
            "schema_version": 1,
            "tasks": [
                {
                    "id": "T1",
                    "title": "Task One",
                    "milestone": "M0",
                    "dependencies": [],
                    "allowed_write_paths": ["scripts/tool.py"],
                    "status": "TODO",
                    "assignee": None,
                    "claim": None,
                },
                {
                    "id": "T2",
                    "title": "Task Two",
                    "milestone": "M0",
                    "dependencies": ["T1"],
                    "allowed_write_paths": ["scripts/other.py"],
                    "status": "TODO",
                    "assignee": None,
                    "claim": None,
                },
                {
                    "id": "T3",
                    "title": "Task Three Colliding",
                    "milestone": "M0",
                    "dependencies": [],
                    "allowed_write_paths": ["./scripts/tool.py"],  # Aliased path
                    "status": "TODO",
                    "assignee": None,
                    "claim": None,
                },
            ],
            "locks": [],
        }
        self.registry_file.write_text(json.dumps(self.initial_data, indent=2))
        self.mgr = RegistryManager(self.registry_file)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_ready_tasks_initial_state(self) -> None:
        """T1 and T3 are ready (no deps), but T2 is blocked by T1."""
        ready = self.mgr.with_lock(lambda d: (get_ready_tasks(d), False))
        ready_ids = {t["id"] for t in ready}
        self.assertIn("T1", ready_ids)
        self.assertIn("T3", ready_ids)
        self.assertNotIn("T2", ready_ids, "T2 must not be ready until T1 is VERIFIED")

    def test_claim_acquires_lock_and_blocks_collision_and_aliases(self) -> None:
        """Claiming T1 locks 'scripts/tool.py' and blocks T3 due to aliased path collision."""
        target = self.mgr.with_lock(claim_task, "T1", "agent_alpha", "commit_123")
        self.assertEqual(target["status"], "IN_PROGRESS")
        self.assertEqual(target["assignee"], "agent_alpha")

        # Now T3 cannot be claimed because its write path collisions with T1
        with self.assertRaises(TaskError) as ctx:
            self.mgr.with_lock(claim_task, "T3", "agent_beta", "commit_123")
        self.assertIn("Write path collision", str(ctx.exception))

        # T3 should also disappear from ready tasks while T1 is locked
        ready = self.mgr.with_lock(lambda d: (get_ready_tasks(d), False))
        ready_ids = {t["id"] for t in ready}
        self.assertNotIn("T3", ready_ids, "T3 should be excluded from ready due to write collision")

    def test_claim_blocked_dependencies(self) -> None:
        """Attempting to claim T2 before T1 is VERIFIED must fail."""
        with self.assertRaises(TaskError) as ctx:
            self.mgr.with_lock(claim_task, "T2", "agent_alpha")
        self.assertIn("Dependency T1 is not satisfied", str(ctx.exception))

    def test_submit_validates_receipt_file(self) -> None:
        """Submit requires a valid, complete execution receipt with passing checks."""
        self.mgr.with_lock(claim_task, "T1", "agent_alpha")

        # 1. Non-existent receipt fails
        with self.assertRaises(TaskError):
            self.mgr.with_lock(submit_task, "T1", str(self.temp_path / "missing.json"), "agent_alpha")

        # 2. Template-only receipt fails
        bad_receipt = self.temp_path / "bad_receipt.json"
        bad_receipt.write_text(json.dumps({
            "template_only": True,
            "task_id": "T1",
            "commands": [],
        }))
        with self.assertRaises(TaskError) as ctx:
            self.mgr.with_lock(submit_task, "T1", str(bad_receipt), "agent_alpha")
        self.assertIn("template_only=True", str(ctx.exception))

        # 3. Valid receipt passes
        good_receipt = self.temp_path / "good_receipt.json"
        good_receipt.write_text(json.dumps({
            "template_only": False,
            "task_id": "T1",
            "commands": [{"command": "pytest", "exit_code": 0}],
            "requirement_evidence": [{"requirement_id": "T1-C01", "status": "PASS"}],
        }))
        target = self.mgr.with_lock(submit_task, "T1", str(good_receipt), "agent_alpha")
        self.assertEqual(target["status"], "REVIEW")

    def test_review_requires_independent_reviewer(self) -> None:
        """Author cannot self-review their own task without explicit override."""
        self.mgr.with_lock(claim_task, "T1", "agent_alpha")

        good_receipt = self.temp_path / "good_receipt.json"
        good_receipt.write_text(json.dumps({
            "template_only": False,
            "task_id": "T1",
            "commands": [{"command": "pytest", "exit_code": 0}],
            "requirement_evidence": [{"requirement_id": "T1-C01", "status": "PASS"}],
        }))
        self.mgr.with_lock(submit_task, "T1", str(good_receipt), "agent_alpha")

        # Self-review rejected
        with self.assertRaises(TaskError) as ctx:
            self.mgr.with_lock(review_task, "T1", "agent_alpha", "VERIFIED")
        self.assertIn("Independent reviewer required", str(ctx.exception))

        # Independent reviewer accepted -> lock released -> T2 becomes ready!
        target = self.mgr.with_lock(review_task, "T1", "independent_reviewer", "VERIFIED")
        self.assertEqual(target["status"], "VERIFIED")

        ready = self.mgr.with_lock(lambda d: (get_ready_tasks(d), False))
        ready_ids = {t["id"] for t in ready}
        self.assertIn("T2", ready_ids, "T2 must become ready once T1 is VERIFIED")

    def test_release_requires_mandatory_reason(self) -> None:
        """Releasing an in-progress task requires non-empty reason and restores TODO."""
        self.mgr.with_lock(claim_task, "T1", "agent_alpha")

        # Empty reason rejected
        with self.assertRaises(TaskError):
            self.mgr.with_lock(release_task, "T1", "   ")

        # Valid release clears lock and resets to TODO
        target = self.mgr.with_lock(release_task, "T1", "Agent disconnected")
        self.assertEqual(target["status"], "TODO")
        self.assertIsNone(target["assignee"])

        # Locks cleared, T3 can now be claimed
        self.mgr.with_lock(claim_task, "T3", "agent_beta")


if __name__ == "__main__":
    unittest.main()
