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
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if torso_id == -1: return 0.0, 0.0 
    
    w, x, y, z = data.xquat[torso_id]
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x**2 + y**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    return np.degrees(pitch), np.degrees(roll)

# ==========================================
# LEVEL 04: LONG STRIDE & HIGH-ARC THROW
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
        # Note for G1: Hip Pitch Negative = Leg Forward. Knee Positive = Bent.
        self.keyframes = {
            # 0. STARTING STANCE (Athletic ready position)
            0: { 
                'left_hip_pitch_joint': -0.3, 'left_knee_joint': 0.6, 'left_ankle_pitch_joint': -0.2,
                'right_hip_pitch_joint': -0.3, 'right_knee_joint': 0.6, 'right_ankle_pitch_joint': -0.2,
                'right_shoulder_pitch_joint': 0.0, 'right_elbow_joint': 0.5,
                'left_shoulder_pitch_joint': 0.0, 'left_elbow_joint': 0.5,
            },
            
            # --- STEP 1: LONG RIGHT STRIDE ---
            30: {'waist_roll_joint': 0.2, 'left_ankle_roll_joint': -0.15}, # Shift Left
            # High Knee
            60: {'right_hip_pitch_joint': -0.9, 'right_knee_joint': 1.4}, 
            # Reach far forward (straighten the knee to extend the step)
            90: {'right_hip_pitch_joint': -0.6, 'right_knee_joint': 0.1, 'right_ankle_pitch_joint': -0.1}, 
            # Pull body forward (Right foot plants, left foot pushes back)
            120: {'waist_roll_joint': 0.0, 'right_hip_pitch_joint': -0.2, 'right_knee_joint': 0.5, 
                  'left_hip_pitch_joint': 0.2, 'left_knee_joint': 0.5},
            
            # --- STEP 2: LONG LEFT STRIDE ---
            150: {'waist_roll_joint': -0.2, 'right_ankle_roll_joint': 0.15}, # Shift Right
            # High Knee
            180: {'left_hip_pitch_joint': -0.9, 'left_knee_joint': 1.4}, 
            # Reach far forward
            210: {'left_hip_pitch_joint': -0.6, 'left_knee_joint': 0.1, 'left_ankle_pitch_joint': -0.1}, 
            # Square up feet together to prepare for jump/throw
            240: {'waist_roll_joint': 0.0, 'left_hip_pitch_joint': -0.2, 'left_knee_joint': 0.5, 
                  'right_hip_pitch_joint': -0.2, 'right_knee_joint': 0.5},
            
            # --- PREPARE BASKETBALL THROW ---
            # Torso leans back, elbows strictly limited to ~90 deg (1.6 rad) to prevent backwards bending
            260: { 
                'right_shoulder_pitch_joint': -2.8, 'right_elbow_joint': 1.6, 
                'left_shoulder_pitch_joint': -2.8, 'left_elbow_joint': 1.6,
                'waist_pitch_joint': -0.4,
                'left_knee_joint': 0.8, 'right_knee_joint': 0.8 # Squat down for power
            },
            
            # --- EXPLODE & HIGH THROW ---
            # Torso whips forward, knees extend. 
            # Elbow extends to 0.05 (almost straight, but never negative to protect the joint)
            290: { 
                'right_shoulder_pitch_joint': -1.0, 'right_elbow_joint': 0.05,
                'left_shoulder_pitch_joint': -1.0, 'left_elbow_joint': 0.05,
                'waist_pitch_joint': 0.3,
                'left_knee_joint': 0.2, 'right_knee_joint': 0.2
            }
        }
        self.frame_times = sorted(list(self.keyframes.keys()))

    def apply_controls(self):
        t = self.step_count
        
        # Keyframe Interpolation Logic
        prev_t, next_t = self.frame_times[0], self.frame_times[-1]
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
                targets[j] = val_prev + progress * (val_next - val_prev)

        for joint_name, rad_val in targets.items():
            if joint_name in self.actuator_map:
                idx = self.actuator_map[joint_name]
                self.env.data.ctrl[idx] = rad_val

        self.step_count += 1
        
        # EARLY RELEASE FOR HIGH ARC
        # The swing is between 260 and 290. Releasing at 274 ensures the arm 
        # is pointing upward (approx 45 degrees) when the ball is let go.
        return t == 274

    def reset(self):
        self.step_count = 0

# ==========================================

def view_baseline():
    xml_path = str(ROOT / 'assets' / 'scene_throw_LEVEL03.xml')

    env = G1FixedBodyThrowEnv(xml_path=xml_path)
    policy = OptionDBasketballPolicy(env) 
    print(f"Currently playing: {policy.__class__.__name__}")
    
    print("Opening MuJoCo Viewer... Close the window to stop.")
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        episode = 0
        while viewer.is_running():
            env.reset()
            policy.reset() 
            
            # Start robot slightly higher to avoid toe-stubbing on reset
            env.data.qpos[:3] = [0.0, 0.0, 0.85]
            env.data.qvel[:] = 0.0
            mujoco.mj_forward(env.model, env.data)
            
            pelvis_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
            ball_jnt_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "throw_ball_free")
            ball_vel_idx = env.model.jnt_dofadr[ball_jnt_id]
            
            max_pitch, max_roll = 0.0, 0.0
            
            # ==========================================
            # PHASE 1: WALKING & THROWING SEQUENCE
            # ==========================================
            while policy.step_count < 320 and viewer.is_running():
                should_release = policy.apply_controls()
                
                # --- NON-RL ACTIVE GYROSCOPE (Keeps the robot standing!) ---
                pitch, roll = get_torso_tilt(env.model, env.data)
                
                kp = 5.0  # Push back against tilt
                kd = 0.5  # Dampen the wobble
                
                torque_pitch = (0.0 - pitch) * kp
                torque_roll = (0.0 - roll) * kp
                
                if pelvis_id != -1:
                    env.data.xfrc_applied[pelvis_id, 3] = torque_roll
                    env.data.xfrc_applied[pelvis_id, 4] = torque_pitch
                # -----------------------------------------------------------

                if should_release:
                    weld_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_EQUALITY, "hold_throw_ball")
                    if weld_id != -1:
                        env.data.eq_active[weld_id] = 0 

                mujoco.mj_step(env.model, env.data)

                max_pitch = max(max_pitch, abs(pitch))
                max_roll = max(max_roll, abs(roll))

                viewer.sync()
                time.sleep(getattr(env, 'control_dt', 0.02)) 

            # ==========================================
            # PHASE 2: GRAVITY & BOUNCING LOOP
            # ==========================================
            if viewer.is_running():
                print("Shot released! Letting the ball fly...")
                control_dt = getattr(env, 'control_dt', 0.02)
                physics_dt = env.model.opt.timestep
                substeps = max(1, int(round(control_dt / physics_dt)))
                
                for _ in range(150): 
                    for _ in range(substeps):
                        if pelvis_id != -1:
                            p, r = get_torso_tilt(env.model, env.data)
                            env.data.xfrc_applied[pelvis_id, 3] = -r * 5.0
                            env.data.xfrc_applied[pelvis_id, 4] = -p * 5.0
                            
                        mujoco.mj_step(env.model, env.data) 
                    
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
            print(f"Max Torso Tilt (Kept low by Active PD Stabilizer): Pitch {max_pitch:.2f}°, Roll {max_roll:.2f}°\n")

            episode += 1

    env.close()

if __name__ == "__main__":
    view_baseline()