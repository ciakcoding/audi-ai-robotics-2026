import sys
import time
import numpy as np
from pathlib import Path
import mujoco
import mujoco.viewer

# Set up the root path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.g1_fixed_body_throw_env import G1FixedBodyThrowEnv

# ==========================================
# BASELINE OPTIONS
# ==========================================
class OptionAZeroPolicy:
    """Option A: Does nothing. The arm relaxes and drops the ball."""
    def __init__(self, action_space):
        self.action_shape = action_space.shape
        
    def predict(self, obs):
        return np.zeros(self.action_shape), None

class OptionBRandomPolicy:
    """Option B: Flails randomly. Proves that learning is better than random luck."""
    def __init__(self, action_space):
        self.action_space = action_space
        
    def predict(self, obs):
        # Sample completely random actions from the environment
        return self.action_space.sample(), None
    
class OptionCSwingPolicy:
    def __init__(self, env, total_swing_steps=34):
        self.action_shape = env.action_space.shape
        self.step_count = 0
        self.total_swing_steps = total_swing_steps
        
        # A shoulder-led throw with neutral wrist joints. Keeping the wrist
        # straight avoids the visibly twisted motion of the earlier fit.
        self.start_actions = np.array([
            0.9318, -0.7911, 0.0491, -0.1425, 0.0, 0.0, 0.0
        ])
        self.end_actions = np.array([
            -1.0, 0.0964, 0.0072, -1.0, 0.0, 0.0, 0.0
        ])

    def predict(self, obs):
        action = np.zeros(self.action_shape)
        
        progress = min(1.0, self.step_count / self.total_swing_steps)
        action[:7] = self.start_actions + progress * (self.end_actions - self.start_actions)
        
        self.step_count += 1
        return action, None

    def reset(self):
        self.step_count = 0

# ==========================================

def view_baseline():
    xml_path = str(ROOT / 'assets' / 'scene_throw.xml')

    with open(xml_path, 'r') as f:
        assert "right_wrist_yaw_link" in f.read(), "Error: The ball is not attached to the right wrist in the XML!"

    env = G1FixedBodyThrowEnv(xml_path=xml_path)
    policy = OptionCSwingPolicy(env) 
    print(f"Currently playing: {policy.__class__.__name__}")
    
    print("Opening MuJoCo Viewer... Close the window to stop.")
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        
        episode = 0
        while viewer.is_running():
            obs, _ = env.reset()
            policy.reset() 
            done = False
            
            # PHASE 1: SCRIPTED NON-LEARNING SWING
            while not done and viewer.is_running():
                action, _ = policy.predict(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                ball_pos = env.data.body("throw_ball").xpos
                target_pos = env.data.body("throw_target").xpos
                
                with viewer.lock():
                    viewer.user_scn.ngeom = 0 
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[0],
                        mujoco.mjtGeom.mjGEOM_CYLINDER, 
                        np.zeros(3), np.zeros(3), np.zeros(9), 
                        np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32) 
                    )
                    mujoco.mjv_connector(
                        viewer.user_scn.geoms[0],
                        mujoco.mjtGeom.mjGEOM_CYLINDER, 
                        0.005, 
                        ball_pos,
                        target_pos
                    )
                    viewer.user_scn.ngeom = 1 

                viewer.sync()
                time.sleep(getattr(env, 'control_dt', 0.02)) 
            
            # The environment stops at first ground contact. Do not advance an
            # additional 1.5 seconds after landing, because that only displays
            # post-task rolling and obscures the measured landing point.
            if viewer.is_running() and not info.get("landed", False):
                print("Episode finished. Letting the ball drop to the floor...")
                
                # FIX: Calculate exact micro-steps to prevent the ball from tunnelling through the floor
                control_dt = getattr(env, 'control_dt', 0.02)
                physics_dt = env.model.opt.timestep
                substeps = max(1, int(round(control_dt / physics_dt)))
                
                for _ in range(75): # Watch it drop for 1.5 seconds
                    for _ in range(substeps):
                        mujoco.mj_step(env.model, env.data) # High frequency physics update
                        
                    ball_pos = env.data.body("throw_ball").xpos
                    target_pos = env.data.body("throw_target").xpos
                    
                    with viewer.lock():
                        mujoco.mjv_connector(
                            viewer.user_scn.geoms[0],
                            mujoco.mjtGeom.mjGEOM_CYLINDER, 
                            0.005, 
                            ball_pos,
                            target_pos
                        )
                    
                    viewer.sync()
                    time.sleep(control_dt)
            
            if not viewer.is_running():
                break 
            
            landing_error = info.get("landing_error_xy")
            print(f"First-contact landing error: {landing_error:.3f}m\n" if landing_error is not None else "No landing recorded.\n")
            episode += 1

    env.close()

if __name__ == "__main__":
    view_baseline()
