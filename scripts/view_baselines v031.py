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
    def __init__(self, env, total_swing_steps=40): 
        self.action_shape = env.action_space.shape
        self.step_count = 0
        self.total_swing_steps = total_swing_steps
        
        self.shoulder_idx = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_shoulder_pitch_joint")
        if self.shoulder_idx == -1 or self.shoulder_idx >= self.action_shape[0]:
            self.shoulder_idx = 0 
            
        joint_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "right_shoulder_pitch_joint")
        low_rad = env.model.jnt_range[joint_id][0]
        high_rad = env.model.jnt_range[joint_id][1]
        
        # FIX: Pushing the end angle further back to -135 degrees to achieve a perfectly horizontal arm!
        # (Change this number to -125 or -145 if you want it slightly lower or higher)
        start_rad = np.radians(90.0)
        end_rad = np.radians(-270.0) 
        
        self.start_angle = 2.0 * (start_rad - low_rad) / (high_rad - low_rad) - 1.0
        self.end_angle = 2.0 * (end_rad - low_rad) / (high_rad - low_rad) - 1.0

    def predict(self, obs):
        action = np.zeros(self.action_shape)
        
        progress = min(1.0, self.step_count / self.total_swing_steps)
        current_shoulder_target = self.start_angle + progress * (self.end_angle - self.start_angle)
        
        action[self.shoulder_idx] = current_shoulder_target
        
        if len(action) > 7 and progress >= 1.0:
            action[-1] = 1.0  
            
        self.step_count += 1
        return action, None

    def reset(self):
        self.step_count = 0

# ==========================================

def view_baseline():
    xml_path = str(ROOT / 'assets' / 'unitree_g1' / 'scene_throw.xml')

    with open(xml_path, 'r') as f:
        assert "right_hand_middle_0_link" in f.read(), "Error: The ball is not attached to the hand in the XML!"

    env = G1FixedBodyThrowEnv(xml_path=xml_path)
    policy = OptionCSwingPolicy(env) 
    print(f"Currently playing: {policy.__class__.__name__}")
    
    print("Opening MuJoCo Viewer... Close the window to stop.")
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        
        for episode in range(5): 
            obs, _ = env.reset()
            policy.reset() 
            done = False
            
            target_body_id = env.model.body("throw_target").id
            env.model.body_pos[target_body_id] = [0.55, 0.0, 0.0]
            
            # PHASE 1: THE REINFORCEMENT LEARNING SWING
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
            
            # PHASE 2: LET GRAVITY DO ITS JOB
            if viewer.is_running():
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
            
            dist = np.linalg.norm(target_pos - ball_pos)
            print(f"Final distance to target: {dist:.3f}m\n")

        if viewer.is_running():
            viewer.close() 

    env.close()

if __name__ == "__main__":
    view_baseline()