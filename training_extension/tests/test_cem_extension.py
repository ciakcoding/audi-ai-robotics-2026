from __future__ import annotations

import unittest

from training_extension.quality_check_cem import (
    DEFAULT_STATE,
    run_checks,
)


class CEMContractTest(unittest.TestCase):
    def test_selected_state_contract_and_smoke(self):
        report = run_checks(DEFAULT_STATE, smoke_episodes=2)
        self.assertEqual(report["status"], "PASS")
        self.assertFalse(report["is_reinforcement_learning"])
        self.assertEqual(report["parameter_count"], 15)
        self.assertEqual(report["backboard_contacts"], 0)
        self.assertEqual(report["falls"], 0)


if __name__ == "__main__":
    unittest.main()
