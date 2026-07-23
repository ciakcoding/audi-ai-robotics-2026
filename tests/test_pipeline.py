import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.ppo_throw_env import PPOThrowEnv


class PipelineContractTest(unittest.TestCase):
    def test_frozen_contract_and_spaces(self):
        env = PPOThrowEnv()
        try:
            self.assertEqual(env.model.nu, 29)
            self.assertEqual(env.action_space.shape, (7,))
            self.assertEqual(env.observation_space.shape, (33,))
            self.assertAlmostEqual(env.model.geom_size[env.ball_geom_id, 0], 0.04)
            self.assertTrue(np.allclose(env.target_pos, [0.55, 0.0, 0.0]))
            self.assertAlmostEqual(env.success_radius, 0.10)
        finally:
            env.close()

    def test_zero_residual_reproduces_successful_baseline(self):
        env = PPOThrowEnv()
        try:
            obs, _ = env.reset(seed=2026)
            terminated = truncated = False
            info = {}
            while not (terminated or truncated):
                obs, _, terminated, truncated, info = env.step(
                    np.zeros(env.action_space.shape, dtype=np.float32)
                )
            self.assertTrue(info["success"])
            self.assertFalse(info["has_fallen"])
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
