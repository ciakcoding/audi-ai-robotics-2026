"""
g1_walk_throw_env.py — Level 3: Walk forward + throw ball
==========================================================
Full-body PPO: legs (12), waist (3), right arm (7) = 22 joints.
Robot must walk forward while throwing the ball at the target.

Based on G1FixedBodyThrowEnv but unlocks the whole body.
"""

from __future__ import annotations
from pathlib import Path
import xml.etree.ElementTree as ET
import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class G1WalkThrowEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        xml_path: str | None = None,
        episode_time: float = 4.0,
        control_dt: float = 0.02,
        action_scale: float = 0.5,
        scripted_release_time: float = 1.65,
        target_pos: tuple = (0.55, 0.0, 0.0),
    ):
        super().__init__()
        if xml_path is None:
            xml_path = str(PROJECT_ROOT / "assets" / "scene_throw.xml")
        self.xml_path = Path(xml_path)

        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)

        self.episode_time = float(episode_time)
        self.control_dt = float(control_dt)
        self.frame_skip = max(1, int(round(self.control_dt / self.model.opt.timestep)))
        self.action_scale = float(action_scale)
        self.scripted_release_time = float(scripted_release_time)
        self.target_pos = np.array(target_pos, dtype=np.float64)

        # ── Key body/geom IDs ──
        self.ball_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "throw_ball")
        self.target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "throw_target")
        self.hold_eq_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "hold_throw_ball")
        self.ball_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "throw_ball_free")
        self.ball_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "throw_ball_geom")
        self.floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.target_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "throw_target_geom")
        self.torso_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.pelvis_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

        self.success_radius = float(self.model.geom_size[self.target_geom_id, 0])
        self.hold_body_id = int(self.model.eq_obj1id[self.hold_eq_id])
        self.ball_qpos_adr = int(self.model.jnt_qposadr[self.ball_joint_id])
        self.ball_qvel_adr = int(self.model.jnt_dofadr[self.ball_joint_id])
        self.hold_relpose = self._load_hold_relpose()

        # ── Find controlled joints: legs + waist + right arm ──
        self.controlled_joint_names = self._find_controlled_joints()
        self.controlled_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in self.controlled_joint_names
        ]
        self.controlled_qpos_adr = np.array(
            [self.model.jnt_qposadr[j] for j in self.controlled_joint_ids]
        )
        self.controlled_qvel_adr = np.array(
            [self.model.jnt_dofadr[j] for j in self.controlled_joint_ids]
        )
        self.controlled_actuator_ids = self._find_actuator_ids()
        self.n_controlled = len(self.controlled_joint_names)

        # ── Spaces ──
        # Action: one per controlled joint
        self.action_space = spaces.Box(-1, 1, shape=(self.n_controlled,), dtype=np.float32)
        # Obs: joint_pos(N) + joint_vel(N) + ball_pos(3) + ball_vel(3) + target(3)
        #      + torso_up(3) + torso_fwd(3) + pelvis_vel(3) + prev_action(N) + released(1) + time_left(1)
        obs_dim = (
            self.n_controlled * 2  # joint positions + velocities
            + 3 + 3                # ball pos + vel
            + 3                    # target pos
            + 3 + 3                # torso up + forward (heading)
            + 3                    # pelvis linear velocity
            + self.n_controlled    # prev action
            + 1 + 1                # released flag + time left
        )
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        # ── Nominal pose ──
        self.nominal_qpos = np.zeros(self.model.nq)
        self.nominal_ctrl = np.zeros(self.model.nu)
        self._init_nominal_pose()

        # ── State ──
        self.step_count = 0
        self.released = False
        self.release_time = None
        self.best_dist = np.inf
        self.landed = False
        self.landing_pos = None
        self.prev_action = np.zeros(self.n_controlled)
        self.total_forward_distance = 0.0
        self.prev_pelvis_x = 0.0

    # ═════════════════════════════════════════════════════════
    #  Joint Discovery
    # ═════════════════════════════════════════════════════════

    def _find_controlled_joints(self):
        """Return joint names for legs, waist, and right arm (22 total)."""
        all_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(self.model.njnt)
        ]
        all_names = [n for n in all_names if n]

        # Legs (12): hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
        leg_patterns = [
            "left_hip_pitch", "left_hip_roll", "left_hip_yaw",
            "left_knee", "left_ankle_pitch", "left_ankle_roll",
            "right_hip_pitch", "right_hip_roll", "right_hip_yaw",
            "right_knee", "right_ankle_pitch", "right_ankle_roll",
        ]
        # Waist (3)
        waist_patterns = ["waist_yaw", "waist_roll", "waist_pitch"]
        # Right arm (7)
        arm_patterns = [
            "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
            "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
        ]

        controlled = []
        for pat in leg_patterns + waist_patterns + arm_patterns:
            matches = [n for n in all_names if pat in n and "joint" in n]
            if matches:
                controlled.append(matches[0])
        return controlled

    def _find_actuator_ids(self):
        ids = []
        for jname in self.controlled_joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            found = -1
            for aid in range(self.model.nu):
                if self.model.actuator_trnid[aid, 0] == jid:
                    found = aid
                    break
            if found < 0:
                raise RuntimeError(f"No actuator for joint {jname}")
            ids.append(found)
        return np.array(ids, dtype=np.int32)

    def _init_nominal_pose(self):
        mujoco.mj_resetData(self.model, self.data)
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        self.nominal_qpos[:] = self.data.qpos[:]
        self.nominal_ctrl[:] = 0
        for aid in range(self.model.nu):
            trnid = self.model.actuator_trnid[aid, 0]
            if trnid >= 0:
                qadr = self.model.jnt_qposadr[trnid]
                if qadr < self.model.nq:
                    self.nominal_ctrl[aid] = self.data.qpos[qadr]

    def _load_hold_relpose(self):
        root = ET.parse(self.xml_path).getroot()
        weld = root.find("./equality/weld[@name='hold_throw_ball']")
        if weld is None:
            raise RuntimeError("Missing hold_throw_ball weld in scene XML.")
        relpose = np.fromstring(
            weld.attrib.get("relpose", "0 0 0 1 0 0 0"), sep=" ", dtype=np.float64
        )
        return relpose

    # ═════════════════════════════════════════════════════════
    #  Reset
    # ═════════════════════════════════════════════════════════

    def _place_ball_in_hand(self):
        hand_pos = self.data.xpos[self.hold_body_id].copy()
        hand_mat = self.data.xmat[self.hold_body_id].reshape(3, 3)
        ball_pos = hand_pos + hand_mat @ self.hold_relpose[:3]
        ball_quat = np.empty(4, dtype=np.float64)
        mujoco.mju_mulQuat(
            ball_quat, self.data.xquat[self.hold_body_id], self.hold_relpose[3:7]
        )
        self.data.qpos[self.ball_qpos_adr : self.ball_qpos_adr + 3] = ball_pos
        self.data.qpos[self.ball_qpos_adr + 3 : self.ball_qpos_adr + 7] = ball_quat
        self.data.qvel[self.ball_qvel_adr : self.ball_qvel_adr + 6] = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[:] = self.nominal_qpos

        self.data.ctrl[:] = self.nominal_ctrl
        # Small random perturbation to controlled joints
        self.data.qpos[self.controlled_qpos_adr] += self.np_random.uniform(
            -0.02, 0.02, self.n_controlled
        )

        if self.hold_eq_id >= 0:
            self.data.eq_active[self.hold_eq_id] = 1

        mujoco.mj_forward(self.model, self.data)
        self._place_ball_in_hand()
        self.model.body_pos[self.target_body_id] = self.target_pos
        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.released = False
        self.release_time = None
        self.best_dist = np.inf
        self.landed = False
        self.landing_pos = None
        self.prev_action = np.zeros(self.n_controlled)
        self.total_forward_distance = 0.0
        self.prev_pelvis_x = float(self.data.xpos[self.pelvis_body_id, 0])

        return self._get_obs(), {}

    # ═════════════════════════════════════════════════════════
    #  Step
    # ═════════════════════════════════════════════════════════

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1, 1)

        # Apply action to controlled joints only
        self.data.ctrl[:] = self.nominal_ctrl
        self.data.ctrl[self.controlled_actuator_ids] = (
            self.nominal_ctrl[self.controlled_actuator_ids]
            + self.action_scale * action
        )

        # Ball release
        t = self.step_count * self.control_dt
        if not self.released and t >= self.scripted_release_time:
            if self.hold_eq_id >= 0:
                self.data.eq_active[self.hold_eq_id] = 0
            self.released = True
            self.release_time = t

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self._update_landing()
        self.step_count += 1

        obs = self._get_obs()
        reward = self._compute_reward(action)

        # Distance tracking
        dist = np.linalg.norm(self._ball_pos() - self.target_pos)
        self.best_dist = min(self.best_dist, dist)
        landing_error = (
            None
            if self.landing_pos is None
            else float(np.linalg.norm(self.landing_pos[:2] - self.target_pos[:2]))
        )

        # Fall detection
        torso_height = float(self.data.xpos[self.torso_body_id, 2])
        torso_up = float(
            np.clip(
                self.data.xmat[self.torso_body_id].reshape(3, 3)[2, 2], -1.0, 1.0
            )
        )
        torso_tilt_deg = float(np.degrees(np.arccos(torso_up)))
        has_fallen = bool(torso_height < 0.50 or torso_tilt_deg > 60.0)

        # Forward distance
        pelvis_x = float(self.data.xpos[self.pelvis_body_id, 0])
        self.total_forward_distance += max(0, pelvis_x - self.prev_pelvis_x)
        self.prev_pelvis_x = pelvis_x

        # Ball went far forward: bonus, but don't terminate
        ball_far = bool(self._ball_pos()[0] > 5.0)

        terminated = bool(self.landed or has_fallen)
        truncated = bool(self.step_count * self.control_dt >= self.episode_time)

        info = {
            "dist_to_target": float(dist),
            "best_dist": float(self.best_dist),
            "released": self.released,
            "release_time": self.release_time,
            "landed": self.landed,
            "landing_pos": None if self.landing_pos is None else self.landing_pos.copy(),
            "landing_error_xy": -1.0 if landing_error is None else landing_error,
            "success": bool(
                landing_error is not None
                and landing_error <= self.success_radius
                and not has_fallen
            ),
            "success_radius": self.success_radius,
            "has_fallen": has_fallen,
            "torso_height_m": torso_height,
            "torso_tilt_deg": torso_tilt_deg,
            "total_forward_distance_m": float(self.total_forward_distance),
        }

        self.prev_action = action.copy()
        return obs, float(reward), terminated, truncated, info

    # ═════════════════════════════════════════════════════════
    #  Reward
    # ═════════════════════════════════════════════════════════

    def _compute_reward(self, action):
        r = 0.0

        # ── 1. Balance: stay roughly upright (death zone only) ──
        torso_up = float(np.clip(
            self.data.xmat[self.torso_body_id].reshape(3, 3)[2, 2], -1.0, 1.0))
        torso_tilt = np.degrees(np.arccos(torso_up))
        if torso_tilt > 20.0:  # only penalize large tilts
            r -= 0.5 * (torso_tilt - 20.0) / 40.0

        # ── 2. Height: keep pelvis near standing height (with deadband) ──
        pelvis_z = float(self.data.xpos[self.pelvis_body_id, 2])
        height_err = abs(pelvis_z - 0.79)
        if height_err > 0.04:  # 4cm tolerance for walking oscillation
            r -= 2.0 * (height_err - 0.04)

        # ── 3. Forward velocity ──
        pelvis_vx = float(self.data.qvel[0])
        desired_speed = 0.3
        r += 0.3 * min(pelvis_vx, desired_speed)
        if pelvis_vx < -0.05:
            r -= 0.5

        # ── 4. Throwing: pre-release velocity shaping ──
        ball_pos = self._ball_pos()
        ball_vel = self._ball_vel()
        to_target = self.target_pos - ball_pos
        dist = np.linalg.norm(to_target) + 1e-6
        direction = to_target / dist
        vtt = np.dot(ball_vel, direction)

        if not self.released:
            r += 0.03 * max(vtt, 0.0)  # pre-release: encourage ball toward target
        else:
            r += 0.5 * np.exp(-2.5 * dist) + 0.02 * max(vtt, 0.0)

        # ── 5. Landing bonus ──
        if self.landed and self.landing_pos is not None:
            error = float(np.linalg.norm(self.landing_pos[:2] - self.target_pos[:2]))
            if error < self.success_radius:
                r += 20.0
                r += 10.0 * (1.0 - error / self.success_radius)
            if error < 0.05:
                r += 10.0

        # ── 6. Long throw bonus ──
        if self._ball_pos()[0] > 5.0:
            r += 5.0

        # ── 7. Survival ──
        r += 0.02

        # ── 8. Regularization ──
        r -= 0.001 * np.linalg.norm(action - self.prev_action)
        r -= 0.0005 * np.linalg.norm(self.data.qvel[self.controlled_qvel_adr])
        r -= 0.0005 * np.linalg.norm(action)

        return r

    # ═════════════════════════════════════════════════════════
    #  Observation
    # ═════════════════════════════════════════════════════════

    def _get_obs(self):
        # Torso orientation in world frame
        torso_mat = self.data.xmat[self.torso_body_id].reshape(3, 3)
        torso_up = torso_mat[:, 2]   # world-frame "up" direction
        torso_fwd = torso_mat[:, 0]  # world-frame "forward" direction (heading)

        # Pelvis linear velocity
        pelvis_vel = self.data.qvel[0:3]

        return np.concatenate(
            [
                self.data.qpos[self.controlled_qpos_adr],
                self.data.qvel[self.controlled_qvel_adr],
                self._ball_pos(),
                self._ball_vel(),
                self.target_pos,
                torso_up,
                torso_fwd,
                pelvis_vel,
                self.prev_action,
                [1.0 if self.released else 0.0],
                [max(0.0, self.episode_time - self.step_count * self.control_dt)],
            ]
        ).astype(np.float32)

    # ═════════════════════════════════════════════════════════
    #  Helpers
    # ═════════════════════════════════════════════════════════

    def _update_landing(self):
        if self.landed or not self.released:
            return
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if {int(contact.geom1), int(contact.geom2)} == {self.ball_geom_id, self.floor_geom_id}:
                self.landed = True
                self.landing_pos = self._ball_pos()
                return

    def _ball_pos(self):
        return self.data.xpos[self.ball_body_id].copy()

    def _ball_vel(self):
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.ball_body_id, vel, 0
        )
        return vel[:3].copy()  # linear velocity (not angular)

    # ═════════════════════════════════════════════════════════
    #  Scripted Walking Baseline
    # ═════════════════════════════════════════════════════════

    def get_baseline_action(self, t: float) -> np.ndarray:
        """
        Scripted walking + throwing baseline.
        Returns normalized action (-1 to 1) for all 22 controlled joints.
        Phase 0-1.65s: walk forward with sinusoidal leg patterns.
        Phase 1.65s+: arm swing to throw (uses same baseline as Level 0/1).
        """
        action = np.zeros(self.n_controlled, dtype=np.float64)

        # Find joint indices in the controlled list
        name_to_idx = {n: i for i, n in enumerate(self.controlled_joint_names)}

        # ── Walking gait parameters ──
        freq = 1.5          # Hz, step frequency
        omega = 2 * np.pi * freq
        hip_amp = 0.4       # hip pitch amplitude
        knee_amp = 0.5      # knee bend amplitude
        hip_roll_amp = 0.1  # lateral balance
        ankle_amp = 0.15    # ankle adjustment
        waist_yaw_amp = 0.1 # counter-rotation

        # ── Legs: alternating sinusoidal pattern ──
        phase_left = omega * t
        phase_right = omega * t + np.pi  # 180° out of phase

        # Left leg
        for side, phase in [("left", phase_left), ("right", phase_right)]:
            hip_pitch = f"{side}_hip_pitch_joint"
            hip_roll = f"{side}_hip_roll_joint"
            hip_yaw = f"{side}_hip_yaw_joint"
            knee = f"{side}_knee_joint"
            ankle_pitch = f"{side}_ankle_pitch_joint"
            ankle_roll = f"{side}_ankle_roll_joint"

            if hip_pitch in name_to_idx:
                action[name_to_idx[hip_pitch]] = hip_amp * np.sin(phase)
            if hip_roll in name_to_idx:
                action[name_to_idx[hip_roll]] = hip_roll_amp * np.cos(phase)
            if hip_yaw in name_to_idx:
                action[name_to_idx[hip_yaw]] = 0.05 * np.sin(phase)
            if knee in name_to_idx:
                # Knee bends during forward swing (when hip is moving forward)
                knee_val = knee_amp * max(0, np.sin(phase))
                action[name_to_idx[knee]] = knee_val
            if ankle_pitch in name_to_idx:
                action[name_to_idx[ankle_pitch]] = ankle_amp * np.sin(phase)
            if ankle_roll in name_to_idx:
                action[name_to_idx[ankle_roll]] = hip_roll_amp * 0.5 * np.cos(phase)

        # ── Waist: slight counter-rotation ──
        for wj in ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]:
            if wj in name_to_idx:
                if "yaw" in wj:
                    action[name_to_idx[wj]] = waist_yaw_amp * np.sin(omega * t * 0.5)
                else:
                    action[name_to_idx[wj]] = 0.0

        # ── Right arm: throwing motion ──
        # Pre-computed in normalized action space (like teammate's ppo_throw_env.py).
        # Arm targets: start = [0.93, -0.79, 0.05, -0.14, 0, 0, 0] rad
        #              end   = [-1.0, 0.10, 0.01, -1.00, 0, 0, 0] rad
        # Convert to action space: action = target / action_scale (since nominal=0)
        arm_start_action = np.array([1.86, -1.58, 0.10, -0.28, 0.0, 0.0, 0.0])
        arm_end_action   = np.array([-2.0, 0.20, 0.02, -2.00, 0.0, 0.0, 0.0])
        # Clip to valid action range [-1, 1] — the extremes will saturate
        arm_start_action = np.clip(arm_start_action, -1.0, 1.0)
        arm_end_action   = np.clip(arm_end_action, -1.0, 1.0)

        arm_joints = [
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint", "right_elbow_joint",
            "right_wrist_roll_joint", "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]

        release_t = self.scripted_release_time
        if t < release_t:
            progress = t / release_t
            arm_action = arm_start_action + progress * (arm_end_action - arm_start_action)
        else:
            arm_action = arm_end_action

        for jname, val in zip(arm_joints, arm_action):
            if jname in name_to_idx:
                action[name_to_idx[jname]] = val

        return np.clip(action, -1.0, 1.0).astype(np.float64)
