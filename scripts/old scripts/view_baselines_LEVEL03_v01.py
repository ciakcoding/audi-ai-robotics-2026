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
        self.step_count = 0
        
        # Cache actuator info to quickly map names to MuJoCo hardware indices
        self.actuator_map = {}
        for i in range(env.model.nu):
            name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name: self.actuator_map[name] = i

    def apply_controls(self):
        """Bypasses Gym and writes directly to the MuJoCo physics motors."""
        t = self.step_count
        targets = {}

        # ---------------------------------------------------------
        # STATE MACHINE: The Timeline of a Basketball Shot
        # ---------------------------------------------------------
        if t < 40:
            # PHASE 1: Step Right Foot 
            targets['right_hip_pitch_joint'] = -0.4
            targets['right_knee_joint'] = 0.8
            targets['right_ankle_pitch_joint'] = -0.4
            
        elif t < 80:
            # PHASE 2: Step Left Foot 
            targets['left_hip_pitch_joint'] = -0.4
            targets['left_knee_joint'] = 0.8
            targets['left_ankle_pitch_joint'] = -0.4
            targets['right_hip_pitch_joint'] = -0.4
            targets['right_knee_joint'] = 0.8
            targets['right_ankle_pitch_joint'] = -0.4
            
        elif t < 130:
            # PHASE 3: Gather & Crouch 
            targets['left_hip_pitch_joint'] = -0.6
            targets['left_knee_joint'] = 1.2
            targets['left_ankle_pitch_joint'] = -0.6
            targets['right_hip_pitch_joint'] = -0.6
            targets['right_knee_joint'] = 1.2
            targets['right_ankle_pitch_joint'] = -0.6

            # Right arm (Shooting arm)
            targets['right_shoulder_pitch_joint'] = -1.8
            targets['right_elbow_joint'] = 2.0
            targets['right_wrist_pitch_joint'] = -0.8

            # Left arm (Guide arm)
            targets['left_shoulder_pitch_joint'] = -1.5
            targets['left_shoulder_roll_joint'] = 0.3
            targets['left_elbow_joint'] = 1.5

        elif t < 150:
            # PHASE 4: Elongate & Shoot! 
            targets['left_hip_pitch_joint'] = 0.0
            targets['left_knee_joint'] = 0.0
            targets['left_ankle_pitch_joint'] = 0.0
            targets['right_hip_pitch_joint'] = 0.0
            targets['right_knee_joint'] = 0.0
            targets['right_ankle_pitch_joint'] = 0.0

            targets['right_shoulder_pitch_joint'] = -2.5
            targets['right_elbow_joint'] = 0.0
            targets['right_wrist_pitch_joint'] = 0.5

        else:
            # PHASE 5: Follow through
            targets['right_shoulder_pitch_joint'] = -2.5
            targets['right_wrist_pitch_joint'] = 0.5

        # Write the targets directly to the hardware motors!
        for joint_name, rad_val in targets.items():
            if joint_name in self.actuator_map:
                idx = self.actuator_map[joint_name]
                self.env.data.ctrl[idx] = rad_val

        self.step_count += 1
        
        # Return True if we have reached the exact release frame
        return t > 145

    def reset(self):
        self.step_count = 0

# ==========================================

def view_baseline():
    xml_path = str(ROOT / 'assets' / 'scene_throw_LEVEL03.xml')

    with open(xml_path, 'r') as f:
        assert "right_wrist_yaw_link" in f.read(), "Error: Ball not attached!"

    # We load the env just to get the model, but we will bypass its restricted step() function
    env = G1FixedBodyThrowEnv(xml_path=xml_path)
    policy = OptionDBasketballPolicy(env) 
    print(f"Currently playing: {policy.__class__.__name__}")
    
    print("Opening MuJoCo Viewer... Close the window to stop.")
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        episode = 0
        while viewer.is_running():
            env.reset()
            policy.reset() 
            
            max_downward_speed = 0.0
            max_impact_force = 0.0
            ball_has_bounced = False
            
            ball_jnt_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "throw_ball_free")
            ball_vel_idx = env.model.jnt_dofadr[ball_jnt_id]
            max_pitch, max_roll = 0.0, 0.0
            
            # ==========================================
            # PHASE 1: SCRIPTED SEQUENCE (Bypassing Gym)
            # ==========================================
            while policy.step_count < 170 and viewer.is_running():
                # Apply targets directly to motors
                should_release = policy.apply_controls()
                
                # Manual Ball Release Mechanism
                if should_release:
                    weld_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_EQUALITY, "hold_throw_ball")
                    if weld_id != -1:
                        env.model.eq_active[weld_id] = 0 # Breaks the weld!

                # Step the physics engine directly
                mujoco.mj_step(env.model, env.data)

                pitch, roll = get_torso_tilt(env.model, env.data)
                max_pitch = max(max_pitch, abs(pitch))
                max_roll = max(max_roll, abs(roll))

                viewer.sync()
                time.sleep(getattr(env, 'control_dt', 0.02)) 

            # ==========================================
            # PHASE 2: GRAVITY & BOUNCING LOOP
            # ==========================================
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

            # ==========================================
            # FINAL REPORT
            # ==========================================
            final_ball_pos = env.data.body("throw_ball").xpos
            # Use data.body.xpos for global position of the hoop target
            final_target_pos = env.data.body("throw_target").xpos
            
            # Check full 3D distance for basketball accuracy (X, Y, and Z)
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