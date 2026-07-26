from __future__ import annotations

import unittest

from training_extension.evaluate_derived_baseline import evaluate
from training_extension.quality_check import run_checks


class DerivedBaselineContractTest(unittest.TestCase):
    def test_static_contract_and_smoke(self):
        report = run_checks(smoke_episodes=1)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["physical_rim_segments"], 16)

    def test_evaluation_schema(self):
        summary, rows = evaluate(episodes=1, seed=98_100)
        self.assertEqual(summary["episodes"], 1)
        self.assertEqual(len(rows), 1)
        self.assertIn("crossing_xy_error", rows[0])
        self.assertEqual(summary["success_radius"], 0.10)


if __name__ == "__main__":
    unittest.main()
