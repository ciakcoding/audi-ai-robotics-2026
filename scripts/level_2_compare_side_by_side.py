#!/usr/bin/env python3
"""Side-by-side: CLEAN vs NOISY — close both windows to exit."""

import sys, time, threading
from pathlib import Path
import numpy as np, mujoco, mujoco.viewer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO
from envs.level_2_robustness_env import G1RobustnessEnv
model = PPO.load(str(ROOT / "outputs" / "models" / "selected" / "best" / "best_model.zip"))


def run(env, label, x):
    obs, _ = env.reset()
    ep = 0
    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        v.cam.azimuth = 140
        v.cam.elevation = -20
        v.cam.distance = 3.0
        try:
            import glfw
            w = v._MjPythonBase__window
            if w:
                glfw.set_window_pos(w, x, 50)
        except Exception:
            pass
        while v.is_running():
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            v.sync()
            time.sleep(env.control_dt)
            if terminated or truncated:
                ep += 1
                err = info.get("landing_error_xy", np.inf)
                ok = "OK" if info.get("success") else "XX"
                print(f"[{label}] Ep{ep:03d} {ok}  error={err:.4f}m")
                obs, _ = env.reset()


if __name__ == "__main__":
    print("CLEAN (left)  vs  NOISY (right)")
    print()
    clean = G1RobustnessEnv(enable_all=False)
    noisy = G1RobustnessEnv(enable_all=True)

    t1 = threading.Thread(target=run, args=(clean, "CLEAN ", 100))
    t2 = threading.Thread(target=run, args=(noisy, "NOISY ", 700))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
