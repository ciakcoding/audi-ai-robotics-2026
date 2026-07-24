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
# LEVEL 03: NO-SCATTO SMOOTH WALK & THROW
# ==========================================
class OptionDBasketballPolicy:
    def __init__(self, env):
        self.env = env
        self.step_count = 0
        
        self.actuator_map = {}
        for i in range(env.model.nu):
            name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name: self.actuator_map[name] = i

        # CHOREOGRAPHY KEYFRAMES 
        # Extremely smooth and slow transitions to avoid any jerking ("scatto")
        self.keyframes = {
            # 0. STARTING STANCE: Arms already UP. 
            # Legs in a stable, slightly bent stance. They will NEVER extend.
            0: { 
                'waist_pitch_joint': 0.15, 'waist_roll_joint': 0.0,
                'left_hip_pitch_joint': -0.4, 'left_knee_joint': 0.8, 'left_ankle_pitch_joint': -0.4,
                'right_hip_pitch_joint': -0.4, 'right_knee_joint': 0.8, 'right_ankle_pitch_joint': -0.4,
                
                'right_shoulder_pitch_joint': -2.6, 'right_elbow_joint': 1.8, 'right_wrist_pitch_joint': -0.5,
                'left_shoulder_pitch_joint': -2.6, 'left_shoulder_roll_joint': 0.5, 'left_elbow_joint': 1.8, 'left_wrist_yaw_joint': 0.5
            },
            
            # 1. SHIFT WEIGHT LEFT 
            30: { 
                'waist_roll_joint': 0.15,
            },
            
            # 2. LIFT RIGHT FOOT (Smooth, controlled step)
            60: { 
                'right_hip_pitch_joint': -0.6, 'right_knee_joint': 1.0, 'right_ankle_pitch_joint': -0.2,
            },
            
            # 3. PLANT RIGHT FOOT FORWARD
            90: { 
                'right_hip_pitch_joint': -0.4, 'right_knee_joint': 0.8, 'right_ankle_pitch_joint': -0.4,
            },

            # 4. SHIFT WEIGHT RIGHT 
            120: { 
                'waist_roll_joint': -0.15, 
            },

            # 5. LIFT LEFT FOOT 
            150: { 
                'left_hip_pitch_joint': -0.6, 'left_knee_joint': 1.0, 'left_ankle_pitch_joint': -0.2,
            },
            
            # 6. PLANT LEFT FOOT (Aligned with right foot)
            180: { 
                'left_hip_pitch_joint': -0.4, 'left_knee_joint': 0.8, 'left_ankle_pitch_joint': -0.4,
            },
            
            # 7. CENTER WEIGHT & FREEZE (Stop completely to stabilize momentum)
            210: { 
                'waist_roll_joint': 0.0,
            },
            
            # 8. HOLD THE STOP (Give the physics engine time to settle completely)
            240: { 
                'waist_roll_joint': 0.0,
            },
            
            # 9. EXTREMELY SMOOTH THROW (No leg movement at all!)
            # We take 60 frames (1.2 seconds) just to move the arms to prevent any scatto.
            300: { 
                'right_shoulder_pitch_joint': -1.4, 'right_elbow_joint': 0.5, 'right_wrist_pitch_joint': 0.3,
                'left_shoulder_pitch_joint': -1.4, 'left_shoulder_roll_joint': 0.2, 'left_elbow_joint': 0.8, 'left_wrist_yaw_joint': 0.5
            }
        }
        self.frame_times = sorted(list(self.keyframes.keys()))

    def apply_controls(self):
        t = self.step_count
        
        # Keyframe Interpolation Logic
        prev_t = self.frame_times[0]
        next_t = self.frame_times[-1]
        for ft in self.frame_times:
            if ft <= t: prev_t = ft
            if ft > t:
                next_t = ft
                break
                
        targets = {}
        if prev_t == next_t:
            targets = self.keyframes[prev_t]
        else:
            progress = (t - prev_t) / (next_t - prev_t)
            prev_dict = self.keyframes[prev_t]
            next_dict = self.keyframes[next_t]
            
            all_joints = set(prev_dict.keys()).union(set(next_dict.keys()))
            for j in all_joints:
                val_prev = prev_dict.get(j, 0.0) 
                val_next = next_dict.get(j, 0.0)
                # Ensure perfectly smooth, linear transition between poses
                targets[j] = val_prev + progress * (val_next - val_prev)

        for joint_name, rad_val in targets.items():
            if joint_name in self.actuator_map:
                idx = self.actuator_map[joint_name]
                self.env.data.ctrl[idx] = rad_val

        self.step_count += 1
        
        # Release the ball gracefully midway through the slow arm swing
        return t == 275 

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
            env.reset()
            policy.reset() 
            
            # Reset the robot's base to [0,0,0.8] to ensure it starts perfectly upright
            env.data.qpos[:3] = [0.0, 0.0, 0.8]
            env.data.qvel[:] = 0.0
            mujoco.mj_forward(env.model, env.data)
            
            max_downward_speed = 0.0
            max_impact_force = 0.0
            ball_has_bounced = False
            
            ball_jnt_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "throw_ball_free")
            ball_vel_idx = env.model.jnt_dofadr[ball_jnt_id]
            max_pitch, max_roll = 0.0, 0.0
            
            # ==========================================
            # PHASE 1: SCRIPTED SEQUENCE 
            # (Loop extended to 320 frames to fit the ultra-smooth sequence)
            # ==========================================
            while policy.step_count < 320 and viewer.is_running():
                should_release = policy.apply_controls()
                
                # Manual Ball Release Mechanism
                if should_release:
                    weld_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_EQUALITY, "hold_throw_ball")
                    if weld_id != -1:
                        env.data.eq_active[weld_id] = 0 

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
            final_target_pos = env.data.body("throw_target").xpos
            
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