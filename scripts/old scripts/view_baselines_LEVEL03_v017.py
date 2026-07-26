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
# LEVEL 10: OVERPOWERED GYRO & GUARANTEED THROW
# ==========================================
class OptionDBasketballPolicy:
    def __init__(self, env):
        self.env = env
        self.step_count = 0
        
        self.actuator_map = {}
        for i in range(env.model.nu):
            name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name: self.actuator_map[name] = i

        self.keyframes = {
            # 0. STARTING STANCE
            0: { 
                'left_hip_pitch_joint': -0.2, 'left_knee_joint': 0.4, 'left_ankle_pitch_joint': -0.2,
                'right_hip_pitch_joint': -0.2, 'right_knee_joint': 0.4, 'right_ankle_pitch_joint': -0.2,
                'right_shoulder_pitch_joint': -0.5, 'right_elbow_joint': 1.0,
                'left_shoulder_pitch_joint': -0.5, 'left_elbow_joint': 1.0,
                'waist_pitch_joint': 0.0 
            },
            
            # --- STEP 1: LONG PHYSICAL STRIDE ---
            50: {'waist_roll_joint': 0.15, 'left_ankle_roll_joint': -0.1},
            
            100: {
                'left_hip_pitch_joint': 0.25, 'left_knee_joint': 0.2, 'left_ankle_pitch_joint': -0.05,
                'right_hip_pitch_joint': -1.0, 'right_knee_joint': 1.4, 'right_ankle_pitch_joint': -0.4
            }, 
            
            150: {
                'right_hip_pitch_joint': -0.5, 'right_knee_joint': 0.1, 'right_ankle_pitch_joint': -0.1,
                'left_hip_pitch_joint': 0.4, 'left_knee_joint': 0.3, 'left_ankle_pitch_joint': 0.1
            }, 
            
            # --- STEP 2: LONG PHYSICAL STRIDE ---
            200: {
                'waist_roll_joint': -0.15, 'right_ankle_roll_joint': 0.1, 
                'right_hip_pitch_joint': 0.0, 'right_knee_joint': 0.2, 
                'left_hip_pitch_joint': 0.1, 'left_knee_joint': 0.6    
            },
            
            250: {
                'right_hip_pitch_joint': 0.25, 'right_knee_joint': 0.2, 'right_ankle_pitch_joint': -0.05,
                'left_hip_pitch_joint': -1.0, 'left_knee_joint': 1.4, 'left_ankle_pitch_joint': -0.4
            }, 
            
            300: {
                'left_hip_pitch_joint': -0.5, 'left_knee_joint': 0.1, 'left_ankle_pitch_joint': -0.1,
                'right_hip_pitch_joint': 0.4, 'right_knee_joint': 0.3, 'right_ankle_pitch_joint': 0.1
            }, 
            
            # --- SQUARE UP & PREPARE ---
            # Smoother, safer stance. No extreme deep squat to avoid launching the robot.
            360: {
                'waist_roll_joint': 0.0, 'right_ankle_roll_joint': 0.0, 'left_ankle_roll_joint': 0.0,
                'left_hip_pitch_joint': -0.2, 'left_knee_joint': 0.5, 'left_ankle_pitch_joint': -0.2,
                'right_hip_pitch_joint': -0.2, 'right_knee_joint': 0.5, 'right_ankle_pitch_joint': -0.2,
                'waist_pitch_joint': 0.0 
            },
            
            # --- WIND UP ---
            400: { 
                'right_shoulder_pitch_joint': -3.1, 'right_elbow_joint': 2.5, 'right_wrist_pitch_joint': -0.7,
                'left_shoulder_pitch_joint': -3.1, 'left_elbow_joint': 2.5, 'left_wrist_pitch_joint': -0.7,
            },
            
            # --- EXPLOSIVE THROW ---
            # Fast arm whip, gentle leg extension to preserve balance.
            415: { 
                'right_shoulder_pitch_joint': -1.0, 'right_elbow_joint': 0.1, 'right_wrist_pitch_joint': 0.6,
                'left_shoulder_pitch_joint': -1.0, 'left_elbow_joint': 0.1, 'left_wrist_pitch_joint': 0.6,
                'left_hip_pitch_joint': -0.1, 'left_knee_joint': 0.2, 'left_ankle_pitch_joint': -0.1, 
                'right_hip_pitch_joint': -0.1, 'right_knee_joint': 0.2, 'right_ankle_pitch_joint': -0.1,
            },
            
            # --- RECOVER BALANCE (Arms come down, stand proudly) ---
            500: {
                'right_shoulder_pitch_joint': 0.0, 'right_elbow_joint': 1.0, 
                'left_shoulder_pitch_joint': 0.0, 'left_elbow_joint': 1.0,
                'left_hip_pitch_joint': -0.2, 'left_knee_joint': 0.4, 'left_ankle_pitch_joint': -0.2,
                'right_hip_pitch_joint': -0.2, 'right_knee_joint': 0.4, 'right_ankle_pitch_joint': -0.2,
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
            
            target_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "throw_target")
            if target_body_id != -1:
                env.model.body_pos[target_body_id][0] = 1.8 
                env.model.body_pos[target_body_id][2] = 1.2 
            
            # Spawn slightly elevated to let the strong PD controller catch it safely
            env.data.qpos[:3] = [0.0, 0.0, 0.81] 
            env.data.qvel[:] = 0.0
            mujoco.mj_forward(env.model, env.data)
            
            pelvis_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
            ball_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "throw_ball")
            
            max_pitch, max_roll = 0.0, 0.0
            last_ball_pos = env.data.xpos[ball_body_id].copy()
            ball_released = False
            ball_has_bounced = False
            ball_crossed_hoop = False
            max_downward_speed = 0.0
            max_impact_force = 0.0
            
            control_dt = getattr(env, 'control_dt', 0.02)
            
            # ==========================================
            # SEAMLESS SINGLE SIMULATION LOOP
            # Loop runs for a long 850 frames to show standing recovery!
            # ==========================================
            while policy.step_count < 850 and viewer.is_running():
                policy.apply_controls()
                
                # --- GUARANTEED RELEASE TIMING ---
                # Forces the hand open perfectly mid-swing (Frame 408)
                if policy.step_count == 408 and not ball_released:
                    weld_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_EQUALITY, "hold_throw_ball")
                    if weld_id != -1:
                        env.data.eq_active[weld_id] = 0 
                        ball_released = True
                        print("\n>>> Ball Released! <<<")

                # --- OVERPOWERED PD GYROSCOPE ---
                pitch, roll = get_torso_tilt(env.model, env.data)
                
                kp = 100.0  # Massively increased stiffness
                kd = 20.0   # Massively increased dampening to kill jitter
                
                torque_pitch = (0.0 - pitch) * kp - (env.data.qvel[4] * kd)
                torque_roll = (0.0 - roll) * kp - (env.data.qvel[3] * kd)
                
                # Increased limit to 200 to allow the gyroscope to fight the heavy swing recoil
                torque_pitch = np.clip(torque_pitch, -200.0, 200.0)
                torque_roll = np.clip(torque_roll, -200.0, 200.0)
                
                if pelvis_id != -1:
                    env.data.xfrc_applied[pelvis_id, 3] = torque_roll
                    env.data.xfrc_applied[pelvis_id, 4] = torque_pitch

                mujoco.mj_step(env.model, env.data)
                
                # --- ANALYTICS & PHYSICS TRACKING ---
                current_ball_pos = env.data.xpos[ball_body_id].copy()
                
                if ball_released:
                    # Falling speed right before bounce
                    vz = env.data.qvel[env.model.jnt_dofadr[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "throw_ball_free")] + 2]
                    if not ball_has_bounced and vz < 0:
                        max_downward_speed = min(max_downward_speed, vz)
                        
                    # Check if ball passes through the Hoop Z-plane
                    hx, hy, hz = env.data.xpos[target_body_id]
                    if last_ball_pos[2] > hz and current_ball_pos[2] <= hz:
                        dist_to_center = np.hypot(current_ball_pos[0] - hx, current_ball_pos[1] - hy)
                        if dist_to_center < 0.20: # Passed within hoop radius!
                            ball_crossed_hoop = True
                    
                    # Check Impact Forces
                    for i in range(env.data.ncon):
                        contact = env.data.contact[i]
                        g1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
                        g2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
                        
                        if g1 == "throw_ball_geom" or g2 == "throw_ball_geom":
                            ball_has_bounced = True
                            c_array = np.zeros(6, dtype=np.float64)
                            mujoco.mj_contactForce(env.model, env.data, i, c_array)
                            max_impact_force = max(max_impact_force, abs(c_array[0]))

                last_ball_pos = current_ball_pos.copy()

                max_pitch = max(max_pitch, abs(pitch))
                max_roll = max(max_roll, abs(roll))

                viewer.sync()
                time.sleep(control_dt) 

            if not viewer.is_running():
                break 

            # FINAL REPORT PRINTING
            final_ball_pos = env.data.body("throw_ball").xpos
            final_target_pos = env.data.body("throw_target").xpos
            final_distance = np.linalg.norm(final_target_pos - final_ball_pos)

            print(f"\n--- EPISODE {episode + 1} BASKETBALL REPORT ---")
            print(f"Ball crossed the basketball hoop: {'YES!' if ball_crossed_hoop else 'NO'}")
            print(f"Final distance to hoop center: {final_distance:.3f}m")
            print(f"Max falling speed at impact: {abs(max_downward_speed):.2f} m/s")
            print(f"Maximum impact force (Rim or Floor): {max_impact_force:.2f} Newtons")
            print(f"Max Torso Tilt (Kept extremely low by strong Gyro): Pitch {max_pitch:.2f}°, Roll {max_roll:.2f}°\n")

            episode += 1

    env.close()

if __name__ == "__main__":
    view_baseline()