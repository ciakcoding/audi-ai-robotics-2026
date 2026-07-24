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
# LEVEL 05: TRUE CoG TRANSLATION & PRO BASKETBALL THROW
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
        # Note for G1: Hip Pitch Negative = Leg Forward. Positive = Leg Backward (Extension).
        self.keyframes = {
            # 0. STARTING STANCE (Athletic, balanced)
            0: { 
                'left_hip_pitch_joint': -0.2, 'left_knee_joint': 0.4, 'left_ankle_pitch_joint': -0.2,
                'right_hip_pitch_joint': -0.2, 'right_knee_joint': 0.4, 'right_ankle_pitch_joint': -0.2,
                'right_shoulder_pitch_joint': -0.5, 'right_elbow_joint': 1.0, # Holding ball comfortably
                'left_shoulder_pitch_joint': -0.5, 'left_elbow_joint': 1.0,
                'waist_pitch_joint': 0.0, 'waist_roll_joint': 0.0
            },
            
            # --- STEP 1: TRANSLATE CoG FORWARD (RIGHT STRIDE) ---
            30: {'waist_roll_joint': 0.08, 'left_ankle_roll_joint': -0.05}, # Gentle shift left
            
            # Lift Right knee high, while extending Left Hip to PUSH the body forward
            60: {
                'left_hip_pitch_joint': 0.1, 'left_knee_joint': 0.1, # <-- Stance leg extends back
                'right_hip_pitch_joint': -0.8, 'right_knee_joint': 1.2
            }, 
            
            # Plant Right foot forward
            90: {'right_hip_pitch_joint': -0.4, 'right_knee_joint': 0.2, 'left_hip_pitch_joint': 0.2}, 
            
            # --- STEP 2: TRANSLATE CoG FORWARD (LEFT STRIDE) ---
            # Pull body over Right foot
            120: {
                'waist_roll_joint': -0.08, 'right_ankle_roll_joint': 0.05, 
                'right_hip_pitch_joint': 0.0, 'right_knee_joint': 0.2, # <-- Right leg now pulls CoG forward
                'left_hip_pitch_joint': 0.4, 'left_knee_joint': 0.5    # Left leg trails behind
            },
            
            # Lift Left knee high, Right Hip extends to push forward
            160: {
                'right_hip_pitch_joint': 0.2, 'right_knee_joint': 0.1, # <-- Stance leg extends back
                'left_hip_pitch_joint': -0.8, 'left_knee_joint': 1.2
            }, 
            
            # Plant Left foot forward
            190: {'left_hip_pitch_joint': -0.4, 'left_knee_joint': 0.2, 'right_hip_pitch_joint': 0.3}, 
            
            # --- SQUARE UP & PREPARE FOR THROW ---
            # Bring right foot forward to meet the left foot. Stance is wide and stable.
            220: {
                'waist_roll_joint': 0.0, 'right_ankle_roll_joint': 0.0, 'left_ankle_roll_joint': 0.0,
                'left_hip_pitch_joint': -0.2, 'left_knee_joint': 0.5, 
                'right_hip_pitch_joint': -0.2, 'right_knee_joint': 0.5
            },
            
            # --- WIND UP (PRO BASKETBALL FORM) ---
            # Torso stays almost perfectly straight (only -0.05 lean).
            # Arms go HIGH above and slightly behind the head. Elbows bend deeply.
            260: { 
                'right_shoulder_pitch_joint': -2.8, 'right_elbow_joint': 2.2, 'right_wrist_pitch_joint': -0.5,
                'left_shoulder_pitch_joint': -2.8, 'left_elbow_joint': 2.2, 'left_wrist_pitch_joint': -0.5,
                'waist_pitch_joint': -0.05,
                'left_knee_joint': 0.8, 'right_knee_joint': 0.8 # Squat for power
            },
            
            # --- EXPLODE & HIGH-ARC SHOOT ---
            # Knees extend. Arms shoot UPWARD and forward (-2.1 rad is ~60 degrees upwards).
            # Wrist flicks forward (0.5 rad).
            290: { 
                'right_shoulder_pitch_joint': -2.1, 'right_elbow_joint': 0.1, 'right_wrist_pitch_joint': 0.5,
                'left_shoulder_pitch_joint': -2.1, 'left_elbow_joint': 0.1, 'left_wrist_pitch_joint': 0.5,
                'waist_pitch_joint': 0.05,
                'left_knee_joint': 0.1, 'right_knee_joint': 0.1 # Jump extension
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
        # Releasing midway through the upward arm thrust yields a beautiful high-arc shot.
        return t == 278

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
                
                # --- STRONGER GYROSCOPE ---
                # Because the robot is now actually shifting its weight forward (CoG translation),
                # the PD gains (kp and kd) have been increased to keep the torso rock solid.
                pitch, roll = get_torso_tilt(env.model, env.data)
                
                kp = 10.0  # Increased push-back force
                kd = 1.0   # Increased dampening
                
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
                            env.data.xfrc_applied[pelvis_id, 3] = -r * 10.0
                            env.data.xfrc_applied[pelvis_id, 4] = -p * 10.0
                            
                        mujoco.mj_step(env.model, env.data) 
                    
                    viewer.sync()
                    time.sleep(control_dt)
            
            if not viewer.is_running():
                break 

            # FINAL REPORT
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