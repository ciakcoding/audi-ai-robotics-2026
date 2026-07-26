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
# LEVEL 06: DYNAMIC PHYSICS CALCULATOR
# ==========================================
class OptionDBasketballPolicy:
    def __init__(self, env):
        self.env = env
        self.step_count = 0
        
        self.actuator_map = {}
        for i in range(env.model.nu):
            name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name: self.actuator_map[name] = i

        # CHOREOGRAPHY KEYFRAMES (Extended, Smoother, Longer Steps)
        self.keyframes = {
            # 0. STARTING STANCE (Athletic)
            0: { 
                'left_hip_pitch_joint': -0.2, 'left_knee_joint': 0.4, 'left_ankle_pitch_joint': -0.2,
                'right_hip_pitch_joint': -0.2, 'right_knee_joint': 0.4, 'right_ankle_pitch_joint': -0.2,
                'right_shoulder_pitch_joint': -0.5, 'right_elbow_joint': 1.0,
                'left_shoulder_pitch_joint': -0.5, 'left_elbow_joint': 1.0,
                'waist_pitch_joint': 0.1 # Slight forward lean so head doesn't snap back!
            },
            
            # --- STEP 1: LONG RIGHT STRIDE ---
            40: {'waist_roll_joint': 0.1, 'left_ankle_roll_joint': -0.05, 'waist_pitch_joint': 0.15},
            
            # Deep hip extension on left leg to PUSH forward, high knee on right leg
            80: {
                'left_hip_pitch_joint': 0.2, 'left_knee_joint': 0.2, 
                'right_hip_pitch_joint': -0.9, 'right_knee_joint': 1.3
            }, 
            
            # Plant right foot far forward
            120: {
                'right_hip_pitch_joint': -0.5, 'right_knee_joint': 0.1, 'right_ankle_pitch_joint': -0.1,
                'left_hip_pitch_joint': 0.4, 'left_knee_joint': 0.3 # Trailing leg fully extended
            }, 
            
            # --- STEP 2: LONG LEFT STRIDE ---
            160: {
                'waist_roll_joint': -0.1, 'right_ankle_roll_joint': 0.05, 
                'right_hip_pitch_joint': 0.0, 'right_knee_joint': 0.2, 
                'left_hip_pitch_joint': 0.1, 'left_knee_joint': 0.6    
            },
            
            # Deep hip extension on right leg, high knee on left
            200: {
                'right_hip_pitch_joint': 0.2, 'right_knee_joint': 0.2, 
                'left_hip_pitch_joint': -0.9, 'left_knee_joint': 1.3
            }, 
            
            # Plant left foot far forward
            240: {
                'left_hip_pitch_joint': -0.5, 'left_knee_joint': 0.1, 'left_ankle_pitch_joint': -0.1,
                'right_hip_pitch_joint': 0.4, 'right_knee_joint': 0.3
            }, 
            
            # --- SQUARE UP & PREPARE FOR THROW ---
            280: {
                'waist_roll_joint': 0.0, 'right_ankle_roll_joint': 0.0, 'left_ankle_roll_joint': 0.0,
                'left_hip_pitch_joint': -0.3, 'left_knee_joint': 0.6, 
                'right_hip_pitch_joint': -0.3, 'right_knee_joint': 0.6,
                'waist_pitch_joint': 0.1 # Keep lean forward
            },
            
            # --- WIND UP (ARMS HIGHER) ---
            # Arms point straight up and back (-3.1 rad), elbows deeply bent (2.5)
            320: { 
                'right_shoulder_pitch_joint': -3.1, 'right_elbow_joint': 2.5, 'right_wrist_pitch_joint': -0.7,
                'left_shoulder_pitch_joint': -3.1, 'left_elbow_joint': 2.5, 'left_wrist_pitch_joint': -0.7,
                'left_knee_joint': 0.8, 'right_knee_joint': 0.8 # Squat for power
            },
            
            # --- EXPLODE ---
            # Fast whip forward. End with arms pointing high up (-1.2 rad)
            370: { 
                'right_shoulder_pitch_joint': -1.2, 'right_elbow_joint': 0.0, 'right_wrist_pitch_joint': 0.5,
                'left_shoulder_pitch_joint': -1.2, 'left_elbow_joint': 0.0, 'left_wrist_pitch_joint': 0.5,
                'waist_pitch_joint': 0.2,
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
            # Make the target nearer (1.2m away instead of 2.0m) to ensure an easier throw
            target_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "throw_target")
            if target_body_id != -1:
                env.model.body_pos[target_body_id][0] = 1.2
            
            env.data.qpos[:3] = [0.0, 0.0, 0.85]
            env.data.qvel[:] = 0.0
            mujoco.mj_forward(env.model, env.data)
            
            pelvis_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
            ball_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "throw_ball")
            
            max_pitch, max_roll = 0.0, 0.0
            
            # Variables for velocity tracking
            last_ball_pos = env.data.xpos[ball_body_id].copy()
            last_time = env.data.time
            ball_released = False
            
            # ==========================================
            # PHASE 1: WALKING & THROWING SEQUENCE
            # ==========================================
            while policy.step_count < 400 and viewer.is_running() and not ball_released:
                policy.apply_controls()
                
                # --- STRONGER GYROSCOPE (PREVENTS HEAD SNAP) ---
                pitch, roll = get_torso_tilt(env.model, env.data)
                
                # We increased Kd (damping) massively to prevent the torso from whipping backwards
                kp = 15.0  # Push-back force
                kd = 5.0   # High dampening to smooth out the tilt
                
                target_pitch_lean = 2.0 # Force robot to lean forward slightly while moving
                
                torque_pitch = (target_pitch_lean - pitch) * kp - (env.data.qvel[4] * kd)
                torque_roll = (0.0 - roll) * kp - (env.data.qvel[3] * kd)
                
                if pelvis_id != -1:
                    env.data.xfrc_applied[pelvis_id, 3] = torque_roll
                    env.data.xfrc_applied[pelvis_id, 4] = torque_pitch

                mujoco.mj_step(env.model, env.data)

                # --- REAL-TIME PHYSICS PREDICTOR (CALCULATES RELEASE INSTANT) ---
                # Calculate current ball velocity vector
                current_time = env.data.time
                dt = current_time - last_time
                current_ball_pos = env.data.xpos[ball_body_id].copy()
                
                if dt > 0 and policy.step_count > 330: # Only calculate during the forward swing
                    ball_vel = (current_ball_pos - last_ball_pos) / dt
                    vx, vy, vz = ball_vel
                    x0, y0, z0 = current_ball_pos
                    xt = env.data.xpos[target_body_id][0]
                    zt = env.data.xpos[target_body_id][2]
                    
                    # If ball is moving forward and upward fast enough
                    if vx > 1.0 and vz > 1.0:
                        g = 9.81
                        # Solve Quadratic Physics Eq: zt = z0 + vz*t - 0.5*g*t^2
                        a = 0.5 * g
                        b = -vz
                        c = zt - z0
                        discriminant = b**2 - 4*a*c
                        
                        if discriminant >= 0:
                            # Time until ball falls to the height of the hoop
                            t_impact = (-b + np.sqrt(discriminant)) / (2*a)
                            
                            # Where will the ball be at that time?
                            predicted_x = x0 + vx * t_impact
                            
                            # If the predicted landing spot crosses the target hoop... FIRE!
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
                time.sleep(getattr(env, 'control_dt', 0.02)) 

            # ==========================================
            # PHASE 2: GRAVITY & BOUNCING LOOP
            # ==========================================
            if viewer.is_running():
                print("Shot released! Letting the ball fly...")
                control_dt = getattr(env, 'control_dt', 0.02)
                physics_dt = env.model.opt.timestep
                substeps = max(1, int(round(control_dt / physics_dt)))
                
                for _ in range(120): 
                    for _ in range(substeps):
                        if pelvis_id != -1:
                            p, r = get_torso_tilt(env.model, env.data)
                            env.data.xfrc_applied[pelvis_id, 3] = -r * 15.0
                            env.data.xfrc_applied[pelvis_id, 4] = (2.0 - p) * 15.0
                            
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
            print(f"Max Torso Tilt (Stabilized): Pitch {max_pitch:.2f}°, Roll {max_roll:.2f}°\n")

            episode += 1

    env.close()

if __name__ == "__main__":
    view_baseline()