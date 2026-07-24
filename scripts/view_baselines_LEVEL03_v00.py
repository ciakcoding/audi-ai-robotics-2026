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
# MATH HELPERS
# ==========================================
def get_torso_tilt(model, data):
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    if torso_id == -1: return 0.0, 0.0 
    w, x, y, z = data.xquat[torso_id]
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x**2 + y**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    return np.degrees(pitch), np.degrees(roll)

# ==========================================
# LEVEL 03: BASKETBALL STATE MACHINE POLICY
# ==========================================
class OptionDBasketballPolicy:
    def __init__(self, env):
        self.env = env
        self.action_shape = env.action_space.shape
        self.step_count = 0
        
        # Cache actuator info to quickly map names to array indices
        self.actuator_map = {}
        for i in range(env.model.nu):
            name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name: self.actuator_map[name] = i

    def get_normalized_action(self, joint_name, target_rad):
        """Converts a desired real-world radian angle into the [-1, 1] space required by RL."""
        joint_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id == -1: return 0.0
        
        low_rad = self.env.model.jnt_range[joint_id][0]
        high_rad = self.env.model.jnt_range[joint_id][1]
        
        # Clip target to physical limits
        target_rad = np.clip(target_rad, low_rad, high_rad)
        
        # Normalize to [-1, 1]
        return 2.0 * (target_rad - low_rad) / (high_rad - low_rad) - 1.0

    def predict(self, obs):
        action = np.zeros(self.action_shape)
        t = self.step_count
        
        # Dictionary to hold the target angles for this specific frame
        targets = {}

        # ---------------------------------------------------------
        # STATE MACHINE: The Timeline of a Basketball Shot
        # (Assuming 50Hz control: 50 steps = 1.0 seconds)
        # ---------------------------------------------------------
        if t < 40:
            # PHASE 1: Step Right Foot (Quick sliding step)
            targets['right_hip_pitch_joint'] = -0.4
            targets['right_knee_joint'] = 0.8
            targets['right_ankle_pitch_joint'] = -0.4
            
        elif t < 80:
            # PHASE 2: Step Left Foot (Bring left foot forward to align)
            targets['left_hip_pitch_joint'] = -0.4
            targets['left_knee_joint'] = 0.8
            targets['left_ankle_pitch_joint'] = -0.4
            # Keep right foot bent to absorb weight
            targets['right_hip_pitch_joint'] = -0.4
            targets['right_knee_joint'] = 0.8
            targets['right_ankle_pitch_joint'] = -0.4
            
        elif t < 130:
            # PHASE 3: Gather & Crouch (Both hands up, legs crouched deeply)
            targets['left_hip_pitch_joint'] = -0.6
            targets['left_knee_joint'] = 1.2
            targets['left_ankle_pitch_joint'] = -0.6
            targets['right_hip_pitch_joint'] = -0.6
            targets['right_knee_joint'] = 1.2
            targets['right_ankle_pitch_joint'] = -0.6

            # Right arm (Shooting arm) - Elbow bent, wrist cocked back
            targets['right_shoulder_pitch_joint'] = -1.8
            targets['right_elbow_joint'] = 2.0
            targets['right_wrist_pitch_joint'] = -0.8

            # Left arm (Guide arm) - Raised to side of ball
            targets['left_shoulder_pitch_joint'] = -1.5
            targets['left_shoulder_roll_joint'] = 0.3
            targets['left_elbow_joint'] = 1.5

        elif t < 150:
            # PHASE 4: Elongate & Shoot! (Explosive extension)
            # Legs thrust upwards
            targets['left_hip_pitch_joint'] = 0.0
            targets['left_knee_joint'] = 0.0
            targets['left_ankle_pitch_joint'] = 0.0
            targets['right_hip_pitch_joint'] = 0.0
            targets['right_knee_joint'] = 0.0
            targets['right_ankle_pitch_joint'] = 0.0

            # Right arm extends, wrist flicks forward
            targets['right_shoulder_pitch_joint'] = -2.5
            targets['right_elbow_joint'] = 0.0
            targets['right_wrist_pitch_joint'] = 0.5
            
            # Trigger release mechanism on the last few frames of the shot
            if t > 145 and len(action) > 7:
                action[-1] = 1.0  

        else:
            # PHASE 5: Follow through
            targets['right_shoulder_pitch_joint'] = -2.5
            targets['right_wrist_pitch_joint'] = 0.5
            if len(action) > 7:
                action[-1] = 1.0

        # Apply all targets to the action array
        for joint_name, rad_val in targets.items():
            if joint_name in self.actuator_map:
                idx = self.actuator_map[joint_name]
                action[idx] = self.get_normalized_action(joint_name, rad_val)

        self.step_count += 1
        return action, None

    def reset(self):
        self.step_count = 0

# ==========================================

def view_baseline():
    xml_path = str(ROOT / 'assets' / 'scene_throw_LEVEL03.xml')

    with open(xml_path, 'r') as f:
        assert "right_wrist_yaw_link" in f.read(), "Error: Ball not attached!"

    env = G1FixedBodyThrowEnv(xml_path=xml_path)
    policy = OptionDBasketballPolicy(env) 
    print(f"Currently playing: {policy.__class__.__name__}")
    
    print("Opening MuJoCo Viewer... Close the window to stop.")
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        episode = 0
        while viewer.is_running():
            obs, _ = env.reset()
            policy.reset() 
            done = False
            
            max_downward_speed = 0.0
            max_impact_force = 0.0
            ball_has_bounced = False
            
            ball_jnt_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "throw_ball_free")
            ball_vel_idx = env.model.jnt_dofadr[ball_jnt_id]

            max_pitch, max_roll = 0.0, 0.0
            
            # SCRIPTED SEQUENCE
            # (We run this for a fixed 170 steps to ensure the throw finishes before gym cutoff)
            while not done and policy.step_count < 170 and viewer.is_running():
                action, _ = policy.predict(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                pitch, roll = get_torso_tilt(env.model, env.data)
                max_pitch = max(max_pitch, abs(pitch))
                max_roll = max(max_roll, abs(roll))

                viewer.sync()
                time.sleep(getattr(env, 'control_dt', 0.02)) 

            # GRAVITY & BOUNCING LOOP
            if viewer.is_running():
                print("Shot released! Letting the ball fly toward the hoop...")
                
                control_dt = getattr(env, 'control_dt', 0.02)
                physics_dt = env.model.opt.timestep
                substeps = max(1, int(round(control_dt / physics_dt)))
                
                for _ in range(150): 
                    for _ in range(substeps):
                        mujoco.mj_step(env.model, env.data) 
                        
                        # Monitor for bounces (Rim or Floor)
                        z_vel = env.data.qvel[ball_vel_idx + 2]
                        if not ball_has_bounced and z_vel < 0:
                            max_downward_speed = min(max_downward_speed, z_vel)

                        for i in range(env.data.ncon):
                            contact = env.data.contact[i]
                            g1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
                            g2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
                            
                            if "throw_ball_geom" in (g1, g2):
                                ball_has_bounced = True
                                c_array = np.zeros(6, dtype=np.float64)
                                mujoco.mj_contactForce(env.model, env.data, i, c_array)
                                max_impact_force = max(max_impact_force, abs(c_array[0])) 

                    viewer.sync()
                    time.sleep(control_dt)
            
            if not viewer.is_running():
                break 

            # FINAL REPORT
            final_ball_pos = env.data.body("throw_ball").xpos
            # The target sphere inside the basketball hoop
            final_target_pos = env.model.body_pos[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "throw_target")]
            
            # Check 3D distance for basketball accuracy
            final_distance = np.linalg.norm(final_target_pos - final_ball_pos)

            print(f"\n--- EPISODE {episode + 1} BASKETBALL REPORT ---")
            print(f"Final distance to hoop center: {final_distance:.3f}m")
            print(f"Max falling speed at impact: {abs(max_downward_speed):.2f} m/s")
            print(f"Maximum impact force (Rim or Floor): {max_impact_force:.2f} Newtons\n")
            print(f"Max Torso Tilt (Due to lack of RL balance): Pitch {max_pitch:.2f}°, Roll {max_roll:.2f}°\n")

            episode += 1

    env.close()

if __name__ == "__main__":
    view_baseline()