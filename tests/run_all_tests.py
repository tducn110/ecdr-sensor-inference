#!/usr/bin/env python3
"""run_all_tests.py — Central test runner executing all ECDR unit and regression tests."""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    print("=" * 80)
    print("RUNNING ALL ECDR VERIFICATION & REGRESSION TESTS")
    print("=" * 80)

    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(ROOT / "tests"), pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Errors: {len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Overall status: {'PASSED' if result.wasSuccessful() else 'FAILED'}")
    print("=" * 80)

    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
