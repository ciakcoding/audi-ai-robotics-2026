import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.g1_fixed_body_throw_env import G1FixedBodyThrowEnv


def policy_class():
    source = ROOT / "scripts" / "view_baselines v031.py"
    spec = importlib.util.spec_from_file_location("baseline_policy_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.OptionCSwingPolicy


class Task1ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = G1FixedBodyThrowEnv()

    def test_frozen_world_and_robot(self):
        self.assertEqual(self.env.model.nu, 29)
        self.assertAlmostEqual(self.env.model.opt.timestep, 0.002)
        self.assertAlmostEqual(self.env.model.opt.gravity[2], -9.81)
        self.assertAlmostEqual(self.env.model.geom_size[self.env.ball_geom_id, 0], 0.04)
        self.assertAlmostEqual(self.env.model.body_mass[self.env.ball_body_id], 0.05)
        self.assertTrue((self.env.target_pos == [0.55, 0.0, 0.0]).all())
        self.assertAlmostEqual(self.env.success_radius, 0.10)

    def test_scripted_baseline_lands_successfully(self):
        obs, _ = self.env.reset(seed=2026)
        policy = policy_class()(self.env)
        terminated = truncated = False
        info = {}
        while not (terminated or truncated):
            action, _ = policy.predict(obs)
            obs, _, terminated, truncated, info = self.env.step(action)
        self.assertTrue(info["landed"])
        self.assertTrue(info["success"])
        self.assertLessEqual(info["landing_error_xy"], self.env.success_radius)
        self.assertTrue(info["stability_evaluable"])
        self.assertFalse(info["has_fallen"])


if __name__ == "__main__":
    unittest.main()
