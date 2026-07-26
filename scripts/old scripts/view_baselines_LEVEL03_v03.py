import sys
import time
import numpy as np
from pathlib import Path
import mujoco
import mujoco.viewer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.g1_fixed_body_throw_env import G1FixedBodyThrowEnv

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
# LEVEL 03: SMOOTH BASKETBALL STATE MACHINE
# ==========================================
class OptionDBasketballPolicy:
    def __init__(self, env):
        self.env = env
        self.step_count = 0
        
        self.actuator_map = {}
        for i in range(env.model.nu):
            name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name: self.actuator_map[name] = i

        # Define the exact trajectory of the robot using Keyframes.
        # This prevents floor penetration by moving the joints smoothly.
        self.keyframes = {
            0: {}, # Start at default 0 stance
            
            30: { # Step Right Foot Forward & Begin raising arms
                'right_hip_pitch_joint': -0.6, 'right_knee_joint': 1.2, 'right_ankle_pitch_joint': -0.4,
                'right_shoulder_pitch_joint': -0.5, 'left_shoulder_pitch_joint': -0.5
            },
            
            55: { # Plant Right Foot, Lift Left Foot, Move arms into "Gather"
                'right_hip_pitch_joint': -0.2, 'right_knee_joint': 0.4, 'right_ankle_pitch_joint': -0.2,
                'left_hip_pitch_joint': -0.6, 'left_knee_joint': 1.2, 'left_ankle_pitch_joint': -0.4,
                # Basketball Gather form (Right hand under, Left hand guiding on side)
                'right_shoulder_pitch_joint': -1.8, 'right_elbow_joint': 2.2, 'right_wrist_pitch_joint': -0.8,
                'left_shoulder_pitch_joint': -1.6, 'left_shoulder_roll_joint': 0.6, 'left_elbow_joint': 1.8, 'left_wrist_yaw_joint': 0.8
            },
            
            80: { # Plant Left Foot (Feet aligned) & Deep Crouch
                'right_hip_pitch_joint': -0.6, 'right_knee_joint': 1.2, 'right_ankle_pitch_joint': -0.6,
                'left_hip_pitch_joint': -0.6, 'left_knee_joint': 1.2, 'left_ankle_pitch_joint': -0.6,
                # Hold Gather form tightly
                'right_shoulder_pitch_joint': -2.0, 'right_elbow_joint': 2.3, 'right_wrist_pitch_joint': -0.8,
                'left_shoulder_pitch_joint': -1.8, 'left_shoulder_roll_joint': 0.6, 'left_elbow_joint': 1.8, 'left_wrist_yaw_joint': 0.8
            },
            
            100: { # EXPLODE UPWARDS & SHOOT
                'right_hip_pitch_joint': 0.1, 'right_knee_joint': 0.1, 'right_ankle_pitch_joint': 0.1,
                'left_hip_pitch_joint': 0.1, 'left_knee_joint': 0.1, 'left_ankle_pitch_joint': 0.1,
                # Flick right wrist, extend right elbow. Left arm stays as a guide.
                'right_shoulder_pitch_joint': -2.7, 'right_elbow_joint': 0.3, 'right_wrist_pitch_joint': 0.6,
                'left_shoulder_pitch_joint': -1.8, 'left_shoulder_roll_joint': 0.6, 'left_elbow_joint': 1.8, 'left_wrist_yaw_joint': 0.8
            }
        }
        self.frame_times = sorted(list(self.keyframes.keys()))

    def apply_controls(self):
        t = self.step_count
        
        # Find which keyframes we are currently between
        prev_t = self.frame_times[0]
        next_t = self.frame_times[-1]
        for ft in self.frame_times:
            if ft <= t: prev_t = ft
            if ft > t:
                next_t = ft
                break
                
        # Interpolate all joints smoothly between the previous and next keyframe
        targets = {}
        if prev_t == next_t:
            targets = self.keyframes[prev_t]
        else:
            progress = (t - prev_t) / (next_t - prev_t)
            prev_dict = self.keyframes[prev_t]
            next_dict = self.keyframes[next_t]
            
            # Combine all active joints from both frames
            all_joints = set(prev_dict.keys()).union(set(next_dict.keys()))
            for j in all_joints:
                val_prev = prev_dict.get(j, 0.0) # Assume 0 if not explicitly defined
                val_next = next_dict.get(j, 0.0)
                targets[j] = val_prev + progress * (val_next - val_prev)

        # Apply the smoothly interpolated targets to the hardware motors
        for joint_name, rad_val in targets.items():
            if joint_name in self.actuator_map:
                idx = self.actuator_map[joint_name]
                self.env.data.ctrl[idx] = rad_val

        self.step_count += 1
        
        # Release the ball slightly before the arms fully extend for an optimal parabolic arc
        return t > 94 

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
            
            max_downward_speed = 0.0
            max_impact_force = 0.0
            ball_has_bounced = False
            
            ball_jnt_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "throw_ball_free")
            ball_vel_idx = env.model.jnt_dofadr[ball_jnt_id]
            max_pitch, max_roll = 0.0, 0.0
            
            # PHASE 1: SCRIPTED SEQUENCE (110 steps to finish the shot)
            while policy.step_count < 110 and viewer.is_running():
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

            # PHASE 2: GRAVITY & BOUNCING LOOP
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

            # FINAL REPORT
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