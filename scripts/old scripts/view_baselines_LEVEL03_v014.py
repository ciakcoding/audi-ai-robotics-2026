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
# LEVEL 07: TRUE PID STABILIZER & SMOOTH HUMAN GAIT
# ==========================================
class OptionDBasketballPolicy:
    def __init__(self, env):
        self.env = env
        self.step_count = 0
        
        self.actuator_map = {}
        for i in range(env.model.nu):
            name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name: self.actuator_map[name] = i

        # CHOREOGRAPHY KEYFRAMES (Ultra-smooth, longer time between frames)
        self.keyframes = {
            # 0. STARTING STANCE (Feet planted firmly)
            0: { 
                'left_hip_pitch_joint': -0.2, 'left_knee_joint': 0.4, 'left_ankle_pitch_joint': -0.2,
                'right_hip_pitch_joint': -0.2, 'right_knee_joint': 0.4, 'right_ankle_pitch_joint': -0.2,
                'right_shoulder_pitch_joint': -0.5, 'right_elbow_joint': 1.0,
                'left_shoulder_pitch_joint': -0.5, 'left_elbow_joint': 1.0,
                'waist_pitch_joint': 0.1 
            },
            
            # --- STEP 1: LONG RIGHT STRIDE ---
            # Slower weight shift to prevent jerking
            50: {'waist_roll_joint': 0.15, 'left_ankle_roll_joint': -0.1, 'waist_pitch_joint': 0.15},
            
            # Deep hip extension on left leg (pushing back), high knee on right leg (reaching forward)
            100: {
                'left_hip_pitch_joint': 0.25, 'left_knee_joint': 0.2, 
                'right_hip_pitch_joint': -1.0, 'right_knee_joint': 1.4
            }, 
            
            # Plant right foot far forward
            150: {
                'right_hip_pitch_joint': -0.5, 'right_knee_joint': 0.1, 'right_ankle_pitch_joint': -0.1,
                'left_hip_pitch_joint': 0.4, 'left_knee_joint': 0.3 
            }, 
            
            # --- STEP 2: LONG LEFT STRIDE ---
            # Shift weight smoothly to the right foot
            200: {
                'waist_roll_joint': -0.15, 'right_ankle_roll_joint': 0.1, 
                'right_hip_pitch_joint': 0.0, 'right_knee_joint': 0.2, 
                'left_hip_pitch_joint': 0.1, 'left_knee_joint': 0.6    
            },
            
            # Deep hip extension on right leg, high knee on left
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
            # Bring feet together. Squatting deeply here helps absorb the shock of the throw!
            360: {
                'waist_roll_joint': 0.0, 'right_ankle_roll_joint': 0.0, 'left_ankle_roll_joint': 0.0,
                'left_hip_pitch_joint': -0.4, 'left_knee_joint': 0.8, 'left_ankle_pitch_joint': -0.4,
                'right_hip_pitch_joint': -0.4, 'right_knee_joint': 0.8, 'right_ankle_pitch_joint': -0.4,
                'waist_pitch_joint': 0.1 
            },
            
            # --- WIND UP ---
            # Arms point straight up and back. Torso leans slightly forward to counter-balance the arms.
            400: { 
                'right_shoulder_pitch_joint': -3.1, 'right_elbow_joint': 2.5, 'right_wrist_pitch_joint': -0.7,
                'left_shoulder_pitch_joint': -3.1, 'left_elbow_joint': 2.5, 'left_wrist_pitch_joint': -0.7,
                'waist_pitch_joint': 0.2
            },

            # --- EXPLOSIVE THROW (Changed from 450 to 415) ---
            415: { 
                'right_shoulder_pitch_joint': -1.0, 'right_elbow_joint': 0.1, 'right_wrist_pitch_joint': 0.6,
                'left_shoulder_pitch_joint': -1.0, 'left_elbow_joint': 0.1, 'left_wrist_pitch_joint': 0.6,
                'waist_pitch_joint': 0.1,
                'left_knee_joint': 0.1, 'right_knee_joint': 0.1 
            },
            
            # --- CONTROLLED, DAMPENED EXPLOSION ---
            # We take 50 full frames to swing the arms (slower = less violent momentum transfer to torso).
            # Knees extend smoothly to provide upward power.
            450: { 
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
            # Make the target much nearer and lower 
            target_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "throw_target")
            if target_body_id != -1:
                env.model.body_pos[target_body_id][0] = 1.8  # (was 1.2)
                env.model.body_pos[target_body_id][2] = 1.2  # (was 1.0)
            
            # --- 2. FIX THE DROP ---
            # Set Z-height to 0.79m (G1's exact resting height) so feet spawn exactly on the floor!
            env.data.qpos[:3] = [0.0, 0.0, 0.79] 
            env.data.qvel[:] = 0.0
            mujoco.mj_forward(env.model, env.data)
            
            pelvis_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
            ball_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "throw_ball")
            
            max_pitch, max_roll = 0.0, 0.0
            
            last_ball_pos = env.data.xpos[ball_body_id].copy()
            last_time = env.data.time
            ball_released = False
            
            # PID Variables for Torso Stabilization
            integral_pitch = 0.0
            integral_roll = 0.0
            control_dt = getattr(env, 'control_dt', 0.02)
            
            # ==========================================
            # PHASE 1: WALKING & THROWING SEQUENCE
            # ==========================================
            # Expanded to 500 frames to fit the longer, smoother walk
            while policy.step_count < 500 and viewer.is_running() and not ball_released:
                policy.apply_controls()
                
                # --- TRUE PID CONTROLLER ---
                pitch, roll = get_torso_tilt(env.model, env.data)
                
                target_pitch_lean = 2.0 # Force a slight athletic lean
                
                error_pitch = target_pitch_lean - pitch
                error_roll = 0.0 - roll
                
                # Calculate Integral (I)
                integral_pitch += error_pitch * control_dt
                integral_roll += error_roll * control_dt
                
                # PID Gains
                kp = 25.0  # Proportional (Spring force)
                ki = 5.0   # Integral (Long-term correction)
                kd = 8.0   # Derivative (Dampening)
                
                torque_pitch = (kp * error_pitch) + (ki * integral_pitch) - (kd * env.data.qvel[4])
                torque_roll = (kp * error_roll) + (ki * integral_roll) - (kd * env.data.qvel[3])
                
                # --- PREVENT FLOOR PENETRATION ---
                # We clip the torques to a safe range (-60 to 60). 
                # This prevents the controller from applying infinite force that crushes the robot into the ground.
                # Increased from 60 to 120 to handle the recoil of the long throw!
                torque_pitch = np.clip(torque_pitch, -120.0, 120.0)
                torque_roll = np.clip(torque_roll, -120.0, 120.0)
                
                if pelvis_id != -1:
                    env.data.xfrc_applied[pelvis_id, 3] = torque_roll
                    env.data.xfrc_applied[pelvis_id, 4] = torque_pitch

                mujoco.mj_step(env.model, env.data)

                # --- REAL-TIME PHYSICS PREDICTOR (CALCULATES RELEASE INSTANT) ---
                current_time = env.data.time
                dt = current_time - last_time
                current_ball_pos = env.data.xpos[ball_body_id].copy()
                
                # Start predicting only during the forward swing (Frame 400+)
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

                max_pitch = max(max_pitch, abs(pitch))
                max_roll = max(max_roll, abs(roll))

                viewer.sync()
                time.sleep(control_dt) 

            # ==========================================
            # PHASE 2: GRAVITY & BOUNCING LOOP
            # ==========================================
            if viewer.is_running():
                print("Shot released! Letting the ball fly...")
                physics_dt = env.model.opt.timestep
                substeps = max(1, int(round(control_dt / physics_dt)))
                
                for _ in range(120): 
                    for _ in range(substeps):
                        # Keep PID running to stabilize after the throw
                        if pelvis_id != -1:
                            p, r = get_torso_tilt(env.model, env.data)
                            t_p = np.clip((2.0 - p) * 25.0 - (env.data.qvel[4] * 8.0), -120, 120)
                            t_r = np.clip(-r * 25.0 - (env.data.qvel[3] * 8.0), -120, 120)
                            env.data.xfrc_applied[pelvis_id, 3] = t_r
                            env.data.xfrc_applied[pelvis_id, 4] = t_p
                            
                        mujoco.mj_step(env.model, env.data) 
                    
                    viewer.sync()
                    time.sleep(control_dt)
            
            if not viewer.is_running():
                break 

            final_ball_pos = env.data.body("throw_ball").xpos
            final_target_pos = env.data.body("throw_target").xpos
            final_distance = np.linalg.norm(final_target_pos - final_ball_pos)

            print(f"\n--- EPISODE {episode + 1} BASKETBALL REPORT ---")
            print(f"Final distance to hoop center: {final_distance:.3f}m")
            print(f"Max Torso Tilt (Stabilized by PID): Pitch {max_pitch:.2f}°, Roll {max_roll:.2f}°\n")

            episode += 1

    env.close()

if __name__ == "__main__":
    view_baseline()