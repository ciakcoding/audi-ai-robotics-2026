"""Side-by-side: CLEAN (left) vs NOISY (right) for Level 3."""

import sys, time, threading, numpy as np
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


def make_randomizer(base):
    m = base.model
    return {
        "fric": m.dof_frictionloss.copy(),
        "damp": m.dof_damping.copy(),
        "force": m.actuator_forcerange.copy(),
        "solref": m.opt.o_solref.copy(),
        "solimp": m.opt.o_solimp.copy(),
        "target": base.target.copy(),
        "floor": [i for i in range(m.ngeom) if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE],
    }


def apply_noise(base, snap):
    m = base.model
    m.dof_frictionloss[:] = snap["fric"] * np.random.uniform(0.7, 1.3)
    m.dof_damping[:] = snap["damp"] * np.random.uniform(0.7, 1.3)
    m.actuator_forcerange[:] = snap["force"] * np.random.uniform(0.85, 1.0)
    m.opt.o_solref[0] = snap["solref"][0] * np.random.uniform(0.5, 2.0)
    m.opt.o_solimp[0] = snap["solimp"][0] * np.random.uniform(0.5, 2.0)
    for gid in snap["floor"]:
        m.geom_friction[gid, 0] *= np.random.uniform(0.5, 1.5)
    base.target = snap["target"] + np.random.normal(0, 0.03, 3)
    base.target[2] = snap["target"][2]
    mujoco.mj_forward(m, base.data)


def run_viewer(noisy, x_offset):
    shot_env = SACShotParameterEnv()
    v_env = DummyVecEnv([lambda: shot_env])
    v_env = VecNormalize.load(str(FROZEN / "selected_vecnormalize.pkl"), v_env)
    v_env.training = False; v_env.norm_reward = False
    model = PPO.load(str(FROZEN / "selected_model.zip"), env=v_env, device="cpu")
    base = shot_env.base
    snap = make_randomizer(base)

    label = "NOISY" if noisy else "CLEAN"
    ep = 0
    with mujoco.viewer.launch_passive(base.model, base.data) as v:
        try:
            import glfw
            w = v._MjPythonBase__window
            if w: glfw.set_window_pos(w, x_offset, 50)
        except: pass
        while v.is_running():
            v_env.seed(50000 + ep)
            if noisy:
                apply_noise(base, snap)
            obs = v_env.reset()
            residual, _ = model.predict(obs, deterministic=True)
            params = shot_env.expert_parameters + shot_env.parameter_scales * residual[0]
            terminated = truncated = False
            info = {}
            while v.is_running() and not (terminated or truncated):
                action = controller_action(base, params)
                _, _, terminated, truncated, info = base.step(action)
                v.sync()
                time.sleep(0.02)
            if not v.is_running():
                break
            succ = "YES" if info["success"] else "NO"
            err = info.get("crossing_xy_error", 0) * 100
            print(f"[{label}] Ep{ep+1} {succ} err={err:.2f}cm")
            ep += 1
    v_env.close()


if __name__ == "__main__":
    print("CLEAN (left)  vs  NOISY (right)")
    t1 = threading.Thread(target=run_viewer, args=(False, 100))
    t2 = threading.Thread(target=run_viewer, args=(True, 700))
    t1.start(); t2.start()
    t1.join(); t2.join()
