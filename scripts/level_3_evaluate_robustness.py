"""Level 3 robustness: clean vs noisy using frozen PPO parameter policy."""

import sys, json, numpy as np
from pathlib import Path
import mujoco

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from training_extension.optimize_direct import controller_action
from training_extension.sac_parameter_env import SACShotParameterEnv

HERE = Path(__file__).resolve().parent
FROZEN = HERE / "frozen" / "ppo_parameters_12288_selected_20260726"
MODEL_PATH = str(FROZEN / "selected_model.zip")
VECNORM_PATH = str(FROZEN / "selected_vecnormalize.pkl")


class DomainRandomizer:
    """Add physics noise to the env base before each episode."""

    def __init__(self, env_base, enable=True):
        self.env = env_base
        self.enable = enable
        m = env_base.model
        self._frictionloss = m.dof_frictionloss.copy()
        self._damping = m.dof_damping.copy()
        self._forcerange = m.actuator_forcerange.copy()
        self._solref = m.opt.o_solref.copy()
        self._solimp = m.opt.o_solimp.copy()
        self._floor_ids = [i for i in range(m.ngeom) if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE]

    def apply(self):
        if not self.enable:
            self._restore()
            return
        m = self.env.model
        # Jitter all parameters by +/- 30% with different random draws
        m.dof_frictionloss[:] = self._frictionloss * np.random.uniform(0.7, 1.3)
        m.dof_damping[:] = self._damping * np.random.uniform(0.7, 1.3)
        m.actuator_forcerange[:] = self._forcerange * np.random.uniform(0.85, 1.0)
        m.opt.o_solref[0] = self._solref[0] * np.random.uniform(0.5, 2.0)
        m.opt.o_solimp[0] = self._solimp[0] * np.random.uniform(0.5, 2.0)
        for gid in self._floor_ids:
            m.geom_friction[gid, 0] *= np.random.uniform(0.5, 1.5)
        # Jitter target position slightly
        old = self.env.target.copy()
        self.env.target = old + np.random.normal(0, 0.03, 3)
        self.env.target[2] = old[2]  # keep z
        mujoco.mj_forward(m, self.env.data)

    def _restore(self):
        m = self.env.model
        m.dof_frictionloss[:] = self._frictionloss
        m.dof_damping[:] = self._damping
        m.actuator_forcerange[:] = self._forcerange
        m.opt.o_solref[:] = self._solref
        m.opt.o_solimp[:] = self._solimp
        mujoco.mj_forward(m, self.env.data)


def run_episodes(env_base, model, vector_env, shot_env, n, label, noisy=False):
    dr = DomainRandomizer(env_base, enable=noisy)
    errors, successes, falls = [], 0, 0

    for ep in range(n):
        vector_env.seed(50000 + ep)
        dr.apply()  # Randomize BEFORE reset so observation reflects perturbed world
        observation = vector_env.reset()

        residual, _ = model.predict(observation, deterministic=True)
        params = shot_env.expert_parameters + shot_env.parameter_scales * residual[0]

        terminated = truncated = False
        info = {}
        while not (terminated or truncated):
            action = controller_action(env_base, params)
            _, _, terminated, truncated, info = env_base.step(action)

        err = info.get("crossing_xy_error", np.inf)
        if err is not None and not np.isinf(err):
            errors.append(float(err) * 100)
        if info.get("success", False):
            successes += 1
        if info.get("has_fallen", False):
            falls += 1
        if (ep + 1) % 10 == 0:
            m = np.mean(errors) if errors else float('nan')
            print(f"  [{label}] {ep+1}/{n} | err={m:.2f}cm | succ={successes} fall={falls}")

    errs = np.array(errors)
    return {
        "label": label, "episodes": n,
        "mean_error_cm": float(np.mean(errs)) if len(errs) else None,
        "median_error_cm": float(np.median(errs)) if len(errs) else None,
        "min_error_cm": float(np.min(errs)) if len(errs) else None,
        "max_error_cm": float(np.max(errs)) if len(errs) else None,
        "std_error_cm": float(np.std(errs)) if len(errs) else None,
        "success_rate": successes / n * 100,
        "fall_rate": falls / n * 100,
    }


def main():
    print("Loading:", MODEL_PATH)
    shot_env = SACShotParameterEnv()
    vector_env = DummyVecEnv([lambda: shot_env])
    vector_env = VecNormalize.load(VECNORM_PATH, vector_env)
    vector_env.training = False; vector_env.norm_reward = False
    model = PPO.load(MODEL_PATH, env=vector_env, device="cpu")
    base = shot_env.base

    print("\n=== CLEAN ===")
    clean = run_episodes(base, model, vector_env, shot_env, 30, "CLEAN", noisy=False)

    print("\n=== NOISY ===")
    noisy = run_episodes(base, model, vector_env, shot_env, 30, "NOISY", noisy=True)

    print("\n" + "=" * 60)
    print("  LEVEL 3 Sim2Real (PPO 12,288 params)")
    print("=" * 60)
    for key, name in [("mean_error_cm", "Error"), ("success_rate", "Success")]:
        c = clean[key]; n = noisy[key]; d = n - c
        print(f"  {name}: CLEAN={c:.1f} | NOISY={n:.1f} | delta={d:+.1f}")

    out = {"clean": clean, "noisy": noisy}
    p = str(ROOT / "outputs" / "level_3_robustness.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nSaved: {p}")

    vector_env.close()


if __name__ == "__main__":
    main()
