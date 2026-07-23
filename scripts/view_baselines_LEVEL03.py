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
# MATH HELPER: GET TORSO TILT
# ==========================================
def get_torso_tilt(model, data):
    """
    Extracts the torso's absolute Pitch and Roll in degrees using quaternions.
    """
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    if torso_id == -1:
        return 0.0, 0.0 # Fallback if torso name changes
        
    # Get quaternion [w, x, y, z]
    quat = data.xquat[torso_id]
    w, x, y, z = quat
    
    # Calculate Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x**2 + y**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Calculate Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    
    # Return as degrees
    return np.degrees(pitch), np.degrees(roll)

# ==========================================
# BASELINE OPTIONS
# ==========================================
class OptionCSwingPolicy:
    def __init__(self, env, total_swing_steps=34):
        self.action_shape = env.action_space.shape
        self.step_count = 0
        self.total_swing_steps = total_swing_steps
        
        # A shoulder-led throw with neutral wrist joints. Keeping the wrist
        # straight avoids the visibly twisted motion of the earlier fit.
        self.start_actions = np.array([
            0.9318, -0.7911, 0.0491, -0.1425, 0.0, 0.0, 0.0
        ])
        self.end_actions = np.array([
            -1.0, 0.0964, 0.0072, -1.0, 0.0, 0.0, 0.0
        ])

    def predict(self, obs):
        action = np.zeros(self.action_shape)
        
        progress = min(1.0, self.step_count / self.total_swing_steps)
        action[:7] = self.start_actions + progress * (self.end_actions - self.start_actions)
        
        self.step_count += 1
        return action, None

    def reset(self):
        self.step_count = 0

# ==========================================

def view_baseline():
    xml_path = str(ROOT / 'assets' / 'scene_throw.xml')

    with open(xml_path, 'r') as f:
        assert "right_wrist_yaw_link" in f.read(), "Error: The ball is not attached to the right wrist in the XML!"

    env = G1FixedBodyThrowEnv(xml_path=xml_path)
    policy = OptionCSwingPolicy(env) 
    print(f"Currently playing: {policy.__class__.__name__}")
    
    print("Opening MuJoCo Viewer... Close the window to stop.")
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        
        episode = 0
        while viewer.is_running():
            obs, _ = env.reset()
            policy.reset() 
            done = False

            # --- TRACKING VARIABLES ---
            max_downward_speed = 0.0
            max_impact_force = 0.0
            ball_has_bounced = False
            
            ball_jnt_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "throw_ball_free")
            ball_vel_idx = env.model.jnt_dofadr[ball_jnt_id]

            max_pitch, max_roll = 0.0, 0.0
            release_time = 0.0
            last_unstable_time = 0.0
            
            # ==========================================
            # PHASE 1: SCRIPTED NON-LEARNING SWING
            # ==========================================
            while not done and viewer.is_running():
                action, _ = policy.predict(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # --- METRICS TRACKING ---
                pitch, roll = get_torso_tilt(env.model, env.data)
                max_pitch = max(max_pitch, abs(pitch))
                max_roll = max(max_roll, abs(roll))
                
                if policy.step_count == policy.total_swing_steps:
                    release_time = env.data.time
                    
                if abs(pitch) > 2.0 or abs(roll) > 2.0:
                    last_unstable_time = env.data.time

                # --- PHYSICS TRACKING ---
                z_vel = env.data.qvel[ball_vel_idx + 2]
                if not ball_has_bounced and z_vel < 0:
                    max_downward_speed = min(max_downward_speed, z_vel)

                for i in range(env.data.ncon):
                    contact = env.data.contact[i]
                    g1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
                    g2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
                    
                    if (g1 == "throw_ball_geom" and g2 == "floor") or (g2 == "throw_ball_geom" and g1 == "floor"):
                        ball_has_bounced = True
                        c_array = np.zeros(6, dtype=np.float64)
                        mujoco.mj_contactForce(env.model, env.data, i, c_array)
                        max_impact_force = max(max_impact_force, abs(c_array[0])) 
                # ------------------------
                
                ball_pos = env.data.body("throw_ball").xpos
                target_pos = env.data.body("throw_target").xpos
                
                with viewer.lock():
                    viewer.user_scn.ngeom = 0 
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[0],
                        mujoco.mjtGeom.mjGEOM_CYLINDER, 
                        np.zeros(3), np.zeros(3), np.zeros(9), 
                        np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32) 
                    )
                    mujoco.mjv_connector(
                        viewer.user_scn.geoms[0],
                        mujoco.mjtGeom.mjGEOM_CYLINDER, 
                        0.005, 
                        ball_pos,
                        target_pos
                    )
                    viewer.user_scn.ngeom = 1 

                viewer.sync()
                time.sleep(getattr(env, 'control_dt', 0.02)) 

            # ==========================================
            # PHASE 2: GRAVITY AND ROLLING LOOP
            # ==========================================
            if viewer.is_running():
                print("Episode finished. Letting the ball bounce and roll...")
                
                control_dt = getattr(env, 'control_dt', 0.02)
                physics_dt = env.model.opt.timestep
                substeps = max(1, int(round(control_dt / physics_dt)))
                
                # CHANGED: Increased from 75 to 150 to give it 3 full seconds to stop rolling
                for _ in range(150): 
                    for _ in range(substeps):
                        mujoco.mj_step(env.model, env.data) 

                        # --- PHYSICS TRACKING (INSIDE SUBSTEP FOR ACCURACY) ---
                        z_vel = env.data.qvel[ball_vel_idx + 2]
                        if not ball_has_bounced and z_vel < 0:
                            max_downward_speed = min(max_downward_speed, z_vel)

                        for i in range(env.data.ncon):
                            contact = env.data.contact[i]
                            g1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
                            g2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
                            
                            if (g1 == "throw_ball_geom" and g2 == "floor") or (g2 == "throw_ball_geom" and g1 == "floor"):
                                ball_has_bounced = True
                                c_array = np.zeros(6, dtype=np.float64)
                                mujoco.mj_contactForce(env.model, env.data, i, c_array)
                                max_impact_force = max(max_impact_force, abs(c_array[0])) 
                        # ------------------------------------------------------

                    # --- METRICS TRACKING ---
                    pitch, roll = get_torso_tilt(env.model, env.data)
                    max_pitch = max(max_pitch, abs(pitch))
                    max_roll = max(max_roll, abs(roll))
                    
                    if abs(pitch) > 2.0 or abs(roll) > 2.0:
                        last_unstable_time = env.data.time
                    # ------------------------
                        
                    ball_pos = env.data.body("throw_ball").xpos
                    target_pos = env.data.body("throw_target").xpos
                    
                    with viewer.lock():
                        mujoco.mjv_connector(
                            viewer.user_scn.geoms[0],
                            mujoco.mjtGeom.mjGEOM_CYLINDER, 
                            0.005, 
                            ball_pos,
                            target_pos
                        )
                    
                    viewer.sync()
                    time.sleep(control_dt)
            
            if not viewer.is_running():
                break 

            # ==========================================
            # FINAL CONSOLIDATED REPORT
            # ==========================================
            stability_time = max(0.0, last_unstable_time - release_time)
            
            final_ball_pos = env.data.body("throw_ball").xpos
            final_target_pos = env.data.body("throw_target").xpos
            final_distance = np.linalg.norm(final_target_pos[:2] - final_ball_pos[:2])

            print(f"\n--- EPISODE {episode + 1} FINAL REPORT ---")
            
            landing_error = info.get("landing_error_xy")
            print(f"First-contact landing error: {landing_error:.3f}m" if landing_error is not None else "First-contact landing error: No landing recorded.")
            print(f"Final distance after rolling: {final_distance:.3f}m")
            print(f"Max falling speed at impact: {abs(max_downward_speed):.2f} m/s")
            print(f"Maximum impact force: {max_impact_force:.2f} Newtons\n")
            
            print(f"Max Torso Tilt   -> Pitch: {max_pitch:.2f}°, Roll: {max_roll:.2f}°")
            
            if last_unstable_time >= env.data.time - 0.001:
                print(f"Time to Stability -> > {stability_time:.2f}s (Did not settle before episode end)\n")
            else:
                print(f"Time to Stability -> {stability_time:.2f}s after release\n")

            episode += 1

    env.close()

if __name__ == "__main__":
    view_baseline()