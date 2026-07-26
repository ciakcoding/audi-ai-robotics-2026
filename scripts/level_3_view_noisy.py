"""View Level 3 PPO policy WITH Sim2Real perturbations."""

import sys, time, numpy as np
from pathlib import Path
import mujoco, mujoco.viewer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from training_extension.optimize_direct import controller_action
from training_extension.sac_parameter_env import SACShotParameterEnv

HERE = Path(__file__).resolve().parent
FROZEN = HERE / "frozen" / "ppo_parameters_12288_selected_20260726"

shot_env = SACShotParameterEnv()
vector_env = DummyVecEnv([lambda: shot_env])
vector_env = VecNormalize.load(str(FROZEN / "selected_vecnormalize.pkl"), vector_env)
vector_env.training = False; vector_env.norm_reward = False
model = PPO.load(str(FROZEN / "selected_model.zip"), env=vector_env, device="cpu")
base = shot_env.base

# Snapshot baselines
m = base.model
fric = m.dof_frictionloss.copy()
damp = m.dof_damping.copy()
force = m.actuator_forcerange.copy()
solref = m.opt.o_solref.copy()
solimp = m.opt.o_solimp.copy()
floor_ids = [i for i in range(m.ngeom) if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE]
target_base = base.target.copy()

print("Level 3 + NOISE — walk + two-hand throw with Sim2Real perturbations")
print("Close window to exit.")

with mujoco.viewer.launch_passive(base.model, base.data) as viewer:
    ep = 0
    while viewer.is_running():
        vector_env.seed(50000 + ep)
        obs = vector_env.reset()

        # Apply random perturbations
        m.dof_frictionloss[:] = fric * np.random.uniform(0.7, 1.3)
        m.dof_damping[:] = damp * np.random.uniform(0.7, 1.3)
        m.actuator_forcerange[:] = force * np.random.uniform(0.85, 1.0)
        m.opt.o_solref[0] = solref[0] * np.random.uniform(0.5, 2.0)
        m.opt.o_solimp[0] = solimp[0] * np.random.uniform(0.5, 2.0)
        for gid in floor_ids:
            m.geom_friction[gid, 0] *= np.random.uniform(0.5, 1.5)
        base.target = target_base + np.random.normal(0, 0.03, 3)
        base.target[2] = target_base[2]
        mujoco.mj_forward(m, base.data)

        residual, _ = model.predict(obs, deterministic=True)
        params = shot_env.expert_parameters + shot_env.parameter_scales * residual[0]

        terminated = truncated = False
        info = {}
        while viewer.is_running() and not (terminated or truncated):
            action = controller_action(base, params)
            _, _, terminated, truncated, info = base.step(action)
            viewer.sync()
            time.sleep(0.02)

        if not viewer.is_running():
            break
        succ = "YES" if info["success"] else "NO"
        err = info.get("crossing_xy_error", 0) * 100
        print(f"Ep{ep+1} {succ} | err={err:.2f}cm | fall={info['has_fallen']}")
        ep += 1

vector_env.close()
