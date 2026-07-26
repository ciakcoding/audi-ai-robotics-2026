import unittest

import numpy as np

from training_extension.basketball_env import BasketballResidualEnv


class TrainingExtensionContractTest(unittest.TestCase):
    def test_frozen_contract(self):
        env = BasketballResidualEnv()
        self.assertEqual(env.model.nu, 29)
        self.assertTrue(np.array_equal(env.target, [2.2, 0.0, 1.2]))
        self.assertEqual(env.hoop_radius, 0.10)
        self.assertEqual(env.release_policy_step, 406)
        self.assertEqual(env.control_substeps, 10)
        self.assertEqual(env.action_space.shape, (24,))
        env.close()

    def test_zero_residual_episode_runs(self):
        env = BasketballResidualEnv()
        obs, _ = env.reset(seed=2026)
        self.assertEqual(obs.shape, env.observation_space.shape)
        terminated = truncated = False
        info = {}
        while not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(
                np.zeros(24, dtype=np.float32)
            )
            self.assertTrue(np.isfinite(reward))
            self.assertTrue(np.all(np.isfinite(obs)))
        self.assertIn("success", info)
        self.assertIn("crossed_hoop_plane", info)
        self.assertIn("touched_backboard", info)
        self.assertIn("hoop_crossing_speed_m_s", info)
        self.assertIn("max_rim_impact_force_n", info)
        self.assertIn("max_torso_tilt_pitch_deg", info)
        self.assertIn("max_torso_tilt_roll_deg", info)
        self.assertIn("max_torso_tilt_yaw_deg", info)
        self.assertIn("ball_to_target_distance_m", info)
        env.close()

    def test_wide_curriculum_uses_virtual_gate(self):
        env = BasketballResidualEnv(curriculum_radius=0.50)
        self.assertTrue(np.all(env.model.geom_contype[env.rim_geom_ids] == 0))
        self.assertEqual(env.curriculum_radius, 0.50)
        env.close()


if __name__ == "__main__":
    unittest.main()
