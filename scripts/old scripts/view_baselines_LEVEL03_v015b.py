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
# LEVEL 08: SEAMLESS CONTROL & PURE PD GYRO
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
        self.keyframes = {
            # 0. STARTING STANCE (Feet planted firmly, pelvis at 0.79m)
            0: { 
                'left_hip_pitch_joint': -0.2, 'left_knee_joint': 0.4, 'left_ankle_pitch_joint': -0.2,
                'right_hip_pitch_joint': -0.2, 'right_knee_joint': 0.4, 'right_ankle_pitch_joint': -0.2,
                'right_shoulder_pitch_joint': -0.5, 'right_elbow_joint': 1.0,
                'left_shoulder_pitch_joint': -0.5, 'left_elbow_joint': 1.0,
                'waist_pitch_joint': 0.1 
            },
            
            # --- STEP 1: LONG PHYSICAL STRIDE ---
            50: {'waist_roll_joint': 0.15, 'left_ankle_roll_joint': -0.1, 'waist_pitch_joint': 0.15},
            
            # Left hip extension (pushing back), high knee on right leg (reaching forward)
            100: {
                'left_hip_pitch_joint': 0.25, 'left_knee_joint': 0.2, 
                'right_hip_pitch_joint': -1.0, 'right_knee_joint': 1.4
            }, 
            
            # Plant right foot far forward
            150: {
                'right_hip_pitch_joint': -0.5, 'right_knee_joint': 0.1, 'right_ankle_pitch_joint': -0.1,
                'left_hip_pitch_joint': 0.4, 'left_knee_joint': 0.3 
            }, 
            
            # --- STEP 2: LONG PHYSICAL STRIDE ---
            200: {
                'waist_roll_joint': -0.15, 'right_ankle_roll_joint': 0.1, 
                'right_hip_pitch_joint': 0.0, 'right_knee_joint': 0.2, 
                'left_hip_pitch_joint': 0.1, 'left_knee_joint': 0.6    
            },
            
            # Right hip extension, high knee on left
            250: {
                'right_hip_pitch_joint': 0.25, 'right_knee_joint': 0.2, 
                'left_hip_pitch_joint': -1.0, 'left_knee_joint': 1.4
            }, 
            
            # Plant left foot far forward
            300: {
                'left_hip_pitch_joint': -0.5, 'left_knee_joint': 0.1, 'left_ankle_pitch_joint': -0.1,
                'right_hip_pitch_joint': 0.4, 'right_knee_joint': 0.3
            }, 
            
            # --- SQUARE UP & DEEP SQUAT ---
            360: {
                'waist_roll_joint': 0.0, 'right_ankle_roll_joint': 0.0, 'left_ankle_roll_joint': 0.0,
                'left_hip_pitch_joint': -0.4, 'left_knee_joint': 0.8, 'left_ankle_pitch_joint': -0.4,
                'right_hip_pitch_joint': -0.4, 'right_knee_joint': 0.8, 'right_ankle_pitch_joint': -0.4,
                'waist_pitch_joint': 0.1 
            },
            
            # --- WIND UP ---
            400: { 
                'right_shoulder_pitch_joint': -3.1, 'right_elbow_joint': 2.5, 'right_wrist_pitch_joint': -0.7,
                'left_shoulder_pitch_joint': -3.1, 'left_elbow_joint': 2.5, 'left_wrist_pitch_joint': -0.7,
                'waist_pitch_joint': 0.2
            },
            
            # --- EXPLOSIVE THROW ---
            # Moves from wind-up to release in 15 frames to generate enough velocity for the 1.8m target!
            415: { 
                'right_shoulder_pitch_joint': -1.0, 'right_elbow_joint': 0.1, 'right_wrist_pitch_joint': 0.6,
                'left_shoulder_pitch_joint': -1.0, 'left_elbow_joint': 0.1, 'left_wrist_pitch_joint': 0.6,
                'waist_pitch_joint': 0.1,
                'left_knee_joint': 0.1, 'right_knee_joint': 0.1 
            }
        }
        self.frame_times = sorted(list(self.keyframes.keys()))

    def apply_controls(self):
        t = self.step_count
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

    def reset(self):
        self.step_count = 0

# ==========================================

def view_baseline():
    xml_path = str(ROOT / 'assets' / 'scene_throw_LEVEL03.xml')

    env = G1FixedBodyThrowEnv(xml_path=xml_path)
    policy = OptionDBasketballPolicy(env) 
    
    print("Opening MuJoCo Viewer... Close the window to stop.")
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        episode = 0
        while viewer.is_running():
            env.reset()
            policy.reset() 
            
            # --- 1. OVERWRITE TARGET POSITION ---
            target_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "throw_target")
            if target_body_id != -1:
                env.model.body_pos[target_body_id][0] = 1.8  # Closer
                env.model.body_pos[target_body_id][2] = 1.2  # Lower
            
            # Spawn perfectly on the floor (no drop)
            env.data.qpos[:3] = [0.0, 0.0, 0.79] 
            env.data.qvel[:] = 0.0
            mujoco.mj_forward(env.model, env.data)
            
            pelvis_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
            ball_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "throw_ball")
            
            max_pitch, max_roll = 0.0, 0.0
            
            last_ball_pos = env.data.xpos[ball_body_id].copy()
            last_time = env.data.time
            ball_released = False
            control_dt = getattr(env, 'control_dt', 0.02)
            
            # ==========================================
            # SEAMLESS SINGLE SIMULATION LOOP
            # ==========================================
            # One clean loop. No breaking or control jumps.
            while policy.step_count < 550 and viewer.is_running():
                # Keeps executing keyframes. After frame 415, the robot 
                # naturally holds its final throw pose.
                policy.apply_controls()
                
                # --- PURE PD GYROSCOPE (NO INTEGRAL WINDUP) ---
                pitch, roll = get_torso_tilt(env.model, env.data)
                target_pitch_lean = 2.0 
                
                # High-performance gains for dynamic transitions
                kp = 45.0  
                kd = 8.0   
                
                torque_pitch = (target_pitch_lean - pitch) * kp - (env.data.qvel[4] * kd)
                torque_roll = (0.0 - roll) * kp - (env.data.qvel[3] * kd)
                
                # Safe limits to prevent floor penetration
                torque_pitch = np.clip(torque_pitch, -120.0, 120.0)
                torque_roll = np.clip(torque_roll, -120.0, 120.0)
                
                if pelvis_id != -1:
                    env.data.xfrc_applied[pelvis_id, 3] = torque_roll
                    env.data.xfrc_applied[pelvis_id, 4] = torque_pitch
                # ----------------------------------------------

                # --- REAL-TIME PHYSICS PREDICTOR ---
                if not ball_released:
                    current_time = env.data.time
                    dt = current_time - last_time
                    current_ball_pos = env.data.xpos[ball_body_id].copy()
                    
                    if dt > 0 and policy.step_count > 400: 
                        ball_vel = (current_ball_pos - last_ball_pos) / dt
                        vx, vy, vz = ball_vel
                        x0, y0, z0 = current_ball_pos
                        xt = env.data.xpos[target_body_id][0]
                        zt = env.data.xpos[target_body_id][2]
                        
                        if vx > 0.5 and vz > 0.5:
                            g = 9.81
                            a = 0.5 * g
                            b = -vz
                            c = zt - z0
                            discriminant = b**2 - 4*a*c
                            
                            if discriminant >= 0:
                                t_impact = (-b + np.sqrt(discriminant)) / (2*a)
                                predicted_x = x0 + vx * t_impact
                                
                                if predicted_x >= xt:
                                    weld_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_EQUALITY, "hold_throw_ball")
                                    if weld_id != -1:
                                        env.data.eq_active[weld_id] = 0 
                                        ball_released = True
                                        print(f"Calculated Release! Vx:{vx:.2f}, Vz:{vz:.2f} | Predicted Land: {predicted_x:.2f}m")

                    last_ball_pos = current_ball_pos
                    last_time = current_time

                mujoco.mj_step(env.model, env.data)

                max_pitch = max(max_pitch, abs(pitch))
                max_roll = max(max_roll, abs(roll))

                viewer.sync()
                time.sleep(control_dt) 

            if not viewer.is_running():
                break 

            final_ball_pos = env.data.body("throw_ball").xpos
            final_target_pos = env.data.body("throw_target").xpos
            final_distance = np.linalg.norm(final_target_pos - final_ball_pos)

            print(f"\n--- EPISODE {episode + 1} BASKETBALL REPORT ---")
            print(f"Final distance to hoop center: {final_distance:.3f}m")
            print(f"Max Torso Tilt (Fully Balanced): Pitch {max_pitch:.2f}°, Roll {max_roll:.2f}°\n")

            episode += 1

    env.close()

if __name__ == "__main__":
    view_baseline()