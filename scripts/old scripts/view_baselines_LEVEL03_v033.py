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
    if torso_id == -1: return 0.0, 0.0, 0.0
    
    w, x, y, z = data.xquat[torso_id]
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x**2 + y**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y**2 + z**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.degrees(pitch), np.degrees(roll), np.degrees(yaw)

# ==========================================
# LEVEL 16: CORRECTED WRISTS & FORWARD ARC
# ==========================================
class OptionDBasketballPolicy:
    def __init__(self, env):
        self.env = env
        self.step_count = 0
        self.actuator_map = {mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(env.model.nu) if mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)}

        self.keyframes = {
            # START
            0: { 
                'left_hip_pitch_joint': -0.2, 'left_knee_joint': 0.4, 'left_ankle_pitch_joint': -0.2,
                'right_hip_pitch_joint': -0.2, 'right_knee_joint': 0.4, 'right_ankle_pitch_joint': -0.2,
                'right_shoulder_pitch_joint': -0.5, 'right_elbow_joint': 1.0,
                'left_shoulder_pitch_joint': -0.5, 'left_elbow_joint': 1.0,
                'waist_pitch_joint': 0.0 
            },
            
            # PHASE 1: PHYSICAL WALK
            50: {'waist_roll_joint': 0.15, 'left_ankle_roll_joint': -0.1},
            100: {
                'left_hip_pitch_joint': 0.25, 'left_knee_joint': 0.2, 'left_ankle_pitch_joint': -0.05,
                'right_hip_pitch_joint': -1.0, 'right_knee_joint': 1.4, 'right_ankle_pitch_joint': -0.4
            }, 
            150: {
                'right_hip_pitch_joint': -0.5, 'right_knee_joint': 0.1, 'right_ankle_pitch_joint': -0.1,
                'left_hip_pitch_joint': 0.4, 'left_knee_joint': 0.3, 'left_ankle_pitch_joint': 0.1
            }, 
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
            
            # PHASE 2: PRO THROW
            350: {
                'waist_roll_joint': 0.0, 'right_ankle_roll_joint': 0.0, 'left_ankle_roll_joint': 0.0,
                'left_hip_pitch_joint': -0.4, 'left_knee_joint': 0.8, 'left_ankle_pitch_joint': -0.4,
                'right_hip_pitch_joint': -0.4, 'right_knee_joint': 0.8, 'right_ankle_pitch_joint': -0.4,
                'waist_pitch_joint': 0.1, 
                'right_shoulder_pitch_joint': -0.5, 'right_elbow_joint': 1.0,
                'left_shoulder_pitch_joint': -0.5, 'left_elbow_joint': 1.0,
            },
            
            # --- THE "SET POINT" ---
            380: { 
                'left_hip_pitch_joint': -0.6, 'left_knee_joint': 1.1, 'left_ankle_pitch_joint': -0.5,
                'right_hip_pitch_joint': -0.6, 'right_knee_joint': 1.1, 'right_ankle_pitch_joint': -0.5,
                
                'right_shoulder_pitch_joint': -1.6, 'right_shoulder_roll_joint': -0.15,
                'right_elbow_joint': 2.4, 'right_wrist_pitch_joint': -1.2, 
                
                'left_shoulder_pitch_joint': -1.5, 'left_shoulder_roll_joint': 0.0,
                'left_elbow_joint': 2.2, 'left_wrist_pitch_joint': -0.2, 'left_wrist_yaw_joint': 0.8,
            },

            # --- UPWARD ACCELERATION ---
            400: {
                'right_shoulder_pitch_joint': -1.8, 'right_elbow_joint': 1.2, 'right_wrist_pitch_joint': -0.6,
                'left_shoulder_pitch_joint': -2.0, 'left_elbow_joint': 1.5, 'left_shoulder_roll_joint': 0.0, 
                'left_wrist_yaw_joint': 0.8,
                'left_knee_joint': 0.4, 'right_knee_joint': 0.4, 
            },

            # --- EXPLOSIVE SNAP (Release & Peel Away) ---
            412: { 
                # Right arm points 45 degrees forward (-1.8) to push the ball to the target, gentle wrist follow through
                'right_shoulder_pitch_joint': -1.8, 'right_elbow_joint': 0.2, 'right_wrist_pitch_joint': 0.2, 
                
                # Left hand completely drops down and straightens out to avoid the head!
                'left_shoulder_pitch_joint': -0.5, 'left_shoulder_roll_joint': 0.5, 'left_elbow_joint': 0.5, 
                'left_wrist_yaw_joint': 0.0, 'left_wrist_pitch_joint': 0.0,
                
                # Full Jump Extension
                'left_hip_pitch_joint': 0.0, 'left_knee_joint': 0.1, 'left_ankle_pitch_joint': 0.0, 
                'right_hip_pitch_joint': 0.0, 'right_knee_joint': 0.1, 'right_ankle_pitch_joint': 0.0,
            },
            
            # --- HOLD THE FOLLOW THROUGH ---
            440: {
                'right_shoulder_pitch_joint': -1.8, 'right_elbow_joint': 0.2, 'right_wrist_pitch_joint': 0.2, 
                'left_shoulder_pitch_joint': -0.5, 'left_elbow_joint': 0.5, 'left_shoulder_roll_joint': 0.5,
                'left_wrist_yaw_joint': 0.0, 'left_wrist_pitch_joint': 0.0,
            },

            # --- RECOVER BALANCE ---
            500: {
                'right_shoulder_pitch_joint': -0.3, 'right_elbow_joint': 0.8, 'right_wrist_pitch_joint': 0.0,
                'left_shoulder_pitch_joint': -0.3, 'left_elbow_joint': 0.8, 'left_wrist_pitch_joint': 0.0,
                'left_shoulder_roll_joint': 0.0, 'right_shoulder_roll_joint': 0.0,
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
            for j in set(self.keyframes[prev_t].keys()).union(set(self.keyframes[next_t].keys())):
                targets[j] = self.keyframes[prev_t].get(j, 0.0) + progress * (self.keyframes[next_t].get(j, 0.0) - self.keyframes[prev_t].get(j, 0.0))

        for joint_name, rad_val in targets.items():
            if joint_name in self.actuator_map:
                # STRICT HUMAN KINEMATICS LIMITS
                if 'elbow' in joint_name: rad_val = max(0.15, rad_val) 
                if 'knee' in joint_name: rad_val = max(0.0, rad_val)
                self.env.data.ctrl[self.actuator_map[joint_name]] = rad_val

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

        import json
        import os
        os.makedirs(str(ROOT / "outputs"), exist_ok=True)
        telemetry = {"time": [], "ball_x": [], "ball_z": [], "pitch": []}

        while viewer.is_running():
            env.reset()
            policy.reset() 
            
            target_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "throw_target")
            if target_body_id != -1:
                env.model.body_pos[target_body_id][0] = 1.8 
                env.model.body_pos[target_body_id][2] = 1.2 
            
            env.data.qpos[:3] = [0.0, 0.0, 0.81] 
            env.data.qvel[:] = 0.0
            mujoco.mj_forward(env.model, env.data)
            
            pelvis_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
            ball_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "throw_ball")
            
            max_pitch, max_roll, max_yaw = 0.0, 0.0, 0.0
            last_ball_pos = env.data.xpos[ball_body_id].copy()
            last_time = env.data.time
            ball_released = False
            
            ball_crossed_hoop = False
            hoop_crossing_speed = 0.0
            max_hoop_impact_force = 0.0
            
            control_dt = getattr(env, 'control_dt', 0.02)
            
            while policy.step_count < 850 and viewer.is_running():
                policy.apply_controls()

                # --- ADVANCED GYRO: ANTI-DRIFT SYSTEM ---
                pitch, roll, yaw = get_torso_tilt(env.model, env.data)
                
                torque_pitch = np.clip((0.0 - pitch) * 100.0 - (env.data.qvel[4] * 20.0), -200.0, 200.0)
                torque_roll = np.clip((0.0 - roll) * 100.0 - (env.data.qvel[3] * 20.0), -200.0, 200.0)
                torque_yaw = np.clip((0.0 - yaw) * 50.0 - (env.data.qvel[5] * 10.0), -100.0, 100.0)
                force_y = np.clip((0.0 - env.data.qpos[1]) * 50.0 - (env.data.qvel[1] * 10.0), -50.0, 50.0)
                
                if pelvis_id != -1:
                    env.data.xfrc_applied[pelvis_id, 3] = torque_roll
                    env.data.xfrc_applied[pelvis_id, 4] = torque_pitch
                    env.data.xfrc_applied[pelvis_id, 5] = torque_yaw
                    env.data.xfrc_applied[pelvis_id, 1] = force_y

                mujoco.mj_step(env.model, env.data)

                # --- RECORD TELEMETRY ---
                telemetry["time"].append(float(env.data.time))
                telemetry["ball_x"].append(float(env.data.body("throw_ball").xpos[0]))
                telemetry["ball_z"].append(float(env.data.body("throw_ball").xpos[2]))
                telemetry["pitch"].append(float(pitch))
                
                # --- AUTO-AIM: REAL-TIME PHYSICS PREDICTOR ---
                current_time = env.data.time
                dt = current_time - last_time if 'last_time' in locals() else control_dt
                current_ball_pos = env.data.xpos[ball_body_id].copy()
                
                if not ball_released and dt > 0 and 395 < policy.step_count < 410:
                    ball_vel = (current_ball_pos - last_ball_pos) / dt
                    vx, vy, vz = ball_vel
                    
                    if vx > 1.0 and vz > 1.0: 
                        x0, y0, z0 = current_ball_pos
                        xt = env.data.xpos[target_body_id][0]
                        zt = env.data.xpos[target_body_id][2]
                        
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
                                    print(f"\n>>> Auto-Aim Triggered! Vx:{vx:.2f}, Vz:{vz:.2f} | Predicted Land: {predicted_x:.2f}m <<<")

                # Failsafe Release
                if policy.step_count == 408 and not ball_released:
                    weld_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_EQUALITY, "hold_throw_ball")
                    if weld_id != -1:
                        env.data.eq_active[weld_id] = 0 
                        ball_released = True
                        print("\n>>> Failsafe Release Triggered! <<<")

                # --- HOOP ANALYTICS TRACKING ---
                if ball_released:
                    hx, hy, hz = env.data.xpos[target_body_id]
                    
                    if last_ball_pos[2] > hz and current_ball_pos[2] <= hz:
                        dist_to_center = np.hypot(current_ball_pos[0] - hx, current_ball_pos[1] - hy)
                        if dist_to_center < 0.15: 
                            ball_crossed_hoop = True
                            if dt > 0:
                                ball_vel = (current_ball_pos - last_ball_pos) / dt
                                hoop_crossing_speed = np.linalg.norm(ball_vel)
                    
                    for i in range(env.data.ncon):
                        contact = env.data.contact[i]
                        g1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
                        g2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
                        
                        if g1 and g2:
                            if ("throw_ball" in g1 and "rim" in g2) or ("throw_ball" in g2 and "rim" in g1):
                                c_array = np.zeros(6, dtype=np.float64)
                                mujoco.mj_contactForce(env.model, env.data, i, c_array)
                                max_hoop_impact_force = max(max_hoop_impact_force, abs(c_array[0]))

                last_ball_pos = current_ball_pos.copy()
                last_time = current_time

                max_pitch = max(max_pitch, abs(pitch))
                max_roll = max(max_roll, abs(roll))
                max_yaw = max(max_yaw, abs(yaw))

                viewer.sync()
                time.sleep(control_dt) 

            if not viewer.is_running():
                break 

            # FINAL REPORT PRINTING
            final_ball_pos = env.data.body("throw_ball").xpos
            final_target_pos = env.data.body("throw_target").xpos
            final_distance = np.linalg.norm(final_target_pos - final_ball_pos)

            print(f"\n--- EPISODE {episode + 1} BASKETBALL REPORT ---")
            print(f"Ball successfully crossed inside the hoop: {'YES!' if ball_crossed_hoop else 'NO (Missed)'}")
            
            if ball_crossed_hoop:
                print(f"Ball Speed exactly at Hoop Crossing: {hoop_crossing_speed:.2f} m/s")
            else:
                print(f"Ball Speed exactly at Hoop Crossing: N/A (Missed the hoop)")
                
            print(f"Maximum impact force on the Hoop Rim: {max_hoop_impact_force:.2f} Newtons")
            print(f"Max Torso Tilt (Anti-Drift + Gyro): Pitch {max_pitch:.2f}°, Roll {max_roll:.2f}°, Yaw {max_yaw:.2f}°\n")

            episode += 1

            # --- SAVE TELEMETRY & EXIT ---
            telemetry["metrics"] = {
                "final_distance": float(final_distance),
                "max_impact_force": float(max_hoop_impact_force)
            }
            with open(str(ROOT / "outputs" / "level03_telemetry.json"), "w") as f:
                json.dump(telemetry, f)
            # break # Uncomment to exit after 1 episode for the plots

    env.close()

if __name__ == "__main__":
    view_baseline()