"""G1 Level-3 concept: official locomotion policy plus a two-hand set shot."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

REUSABLE_PYTHON = Path(r"D:\mujoco_rl_env\python.exe")
try:
    import mujoco
    import numpy as np
    import torch
except ModuleNotFoundError as exc:
    if REUSABLE_PYTHON.exists() and Path(sys.executable).resolve() != REUSABLE_PYTHON.resolve():
        done = subprocess.run(
            [str(REUSABLE_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
            check=False,
        )
        raise SystemExit(done.returncode)
    raise SystemExit(f"Missing dependency: {exc.name}") from exc

ROOT = Path(__file__).resolve().parents[1]
G1_XML = ROOT / "external" / "mujoco_menagerie" / "unitree_g1" / "g1.xml"
POLICY_PATH = (
    ROOT / "references" / "github" / "rl_sar" / "policy" / "g1"
    / "robomimic" / "locomotion" / "policy_29dof.pt"
)

BALL_RADIUS_M = 0.04
BALL_MASS_KG = 0.05
PHYSICS_DT_S = 0.002
CONTROL_DT_S = 0.02
CONTROL_DECIMATION = 10
TARGET_POS_M = np.array([2.50, 0.0, 0.0])
SUCCESS_RADIUS_M = 0.10
FALL_HEIGHT_M = 0.60
FALL_TILT_DEG = 45.0

LEG_NAMES = (
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
)
UPPER_NAMES = (
    "waist_yaw",
    "waist_roll",
    "waist_pitch",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll",
    "left_wrist_pitch",
    "left_wrist_yaw",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
)
DEFAULT_LEGS = np.array(
    [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0]
)
LEG_KP = np.array([100, 100, 100, 150, 40, 40] * 2, dtype=float)
LEG_KD = np.array([2, 2, 2, 4, 2, 2] * 2, dtype=float)
TORQUE_LIMITS = np.array([88, 139, 88, 139, 50, 50] * 2, dtype=float)
JOINT_MAPPING = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23,
     5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=int,
)
POLICY_DEFAULT = np.array(
    [-0.20, -0.20, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.42, 0.42,
     0.35, 0.35, -0.23, -0.23, 0.18, -0.18, 0.0, 0.0, 0.0, 0.0,
     0.87, 0.87, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    dtype=float,
)
POLICY_KP = np.array(
    [200, 200, 200, 150, 150, 200, 150, 150, 200, 200, 200, 100, 100,
     20, 20, 100, 100, 20, 20, 50, 50, 50, 50, 40, 40, 40, 40, 40, 40],
    dtype=float,
)
POLICY_KD = np.array(
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2, 2, 2, 2, 2, 2, 2, 2,
     2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    dtype=float,
)
POLICY_LIMITS = np.array(
    [88, 88, 88, 88, 88, 88, 88, 88, 88, 139, 139, 25, 25, 50, 50,
     25, 25, 50, 50, 25, 25, 25, 25, 5, 5, 5, 5, 5, 5],
    dtype=float,
)


@dataclass
class SequenceResult:
    completed_steps: int
    forward_displacement_m: float
    max_left_foot_clearance_m: float
    max_right_foot_clearance_m: float
    standing_torso_height_m: float
    squat_torso_height_m: float
    recovered_torso_height_m: float
    ball_speed_at_release_mps: float
    landing_x_m: float | None
    landing_y_m: float | None
    landing_error_m: float | None
    min_torso_height_m: float
    max_torso_tilt_deg: float
    fell: bool
    success: bool


def _g1_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(G1_XML))
    spec.delete(spec.key("stand"))
    spec.body("right_wrist_yaw_link").add_site(
        name="throw_hand_site",
        pos=[0.105, 0.039, 0.015],
        size=[0.018],
        rgba=[0.1, 1.0, 0.2, 0.55],
    )
    return spec


def build_model() -> mujoco.MjModel:
    scene = mujoco.MjSpec.from_string(
        f"""
        <mujoco model="g1_official_walk_two_hand_shot">
          <compiler angle="radian"/>
          <option timestep="{PHYSICS_DT_S}" gravity="0 0 -9.81"
                  integrator="implicitfast" solver="Newton" iterations="100"/>
          <visual>
            <headlight diffuse="0.6 0.6 0.6" ambient="0.1 0.1 0.1"
                       specular="0.9 0.9 0.9"/>
            <rgba haze="0.15 0.25 0.35 1"/>
            <global azimuth="140" elevation="-20"/>
          </visual>
          <asset>
            <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7"
                     rgb2="0 0 0" width="512" height="3072"/>
            <texture type="2d" name="groundplane" builtin="checker" mark="edge"
                     rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
                     markrgb="0.8 0.8 0.8" width="300" height="300"/>
            <material name="groundplane" texture="groundplane" texuniform="true"
                      texrepeat="5 5" reflectance="0.2"/>
          </asset>
          <worldbody>
            <geom name="floor" type="plane" size="6 6 0.05"
                  material="groundplane" contype="1" conaffinity="1" condim="3"
                  friction="1 0.005 0.0001"/>
            <body name="throw_target" pos="{TARGET_POS_M[0]} 0 0">
              <geom name="throw_target_geom" type="cylinder"
                    size="{SUCCESS_RADIUS_M} 0.006" contype="0" conaffinity="0"
                    rgba="0.1 0.8 0.25 0.55"/>
            </body>
            <body name="throw_ball" pos="0.35 -0.30 1.20">
              <freejoint name="throw_ball_free"/>
              <geom name="throw_ball_geom" type="sphere" size="{BALL_RADIUS_M}"
                    mass="{BALL_MASS_KG}" contype="1" conaffinity="1" condim="3"
                    rgba="0.9 0.2 0.1 1" friction="1 0.005 0.0001"/>
              <site name="ball_center_site" size="0.005" rgba="1 1 1 0"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    scene.attach(_g1_spec(), prefix="", frame=scene.worldbody.add_frame(name="robot_frame"))
    scene.add_exclude(
        name="exclude_ball_wrist",
        bodyname1="throw_ball",
        bodyname2="right_wrist_yaw_link",
    )
    scene.add_equality(
        name="hold_throw_ball",
        type=mujoco.mjtEq.mjEQ_WELD,
        name1="throw_hand_site",
        name2="ball_center_site",
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        active=1,
    )
    model = scene.compile()
    # Menagerie uses position actuators. The official policy deployment uses
    # torque motors plus explicit PD, so convert the compiled actuators.
    model.actuator_gainprm[:] = 0
    model.actuator_gainprm[:, 0] = 1
    model.actuator_biasprm[:] = 0
    model.actuator_ctrllimited[:] = 0
    return model


class ReferencePolicyWalkShootBaseline:
    def __init__(self) -> None:
        if not POLICY_PATH.exists():
            raise FileNotFoundError(
                f"G1 locomotion policy not found: {POLICY_PATH}. "
                "See references/README.md."
            )
        self.model = build_model()
        self.data = mujoco.MjData(self.model)
        self.policy = torch.jit.load(str(POLICY_PATH), map_location="cpu")
        self.policy.eval()
        self.ids = {}
        for name in (*LEG_NAMES, *UPPER_NAMES):
            joint = self._joint(f"{name}_joint")
            self.ids[name] = (
                self._actuator(f"{name}_joint"),
                int(self.model.jnt_qposadr[joint]),
                int(self.model.jnt_dofadr[joint]),
            )
        self.leg_act = np.array([self.ids[name][0] for name in LEG_NAMES])
        self.leg_qadr = np.array([self.ids[name][1] for name in LEG_NAMES])
        self.leg_dadr = np.array([self.ids[name][2] for name in LEG_NAMES])
        self.pelvis = self._body("pelvis")
        self.torso = self._body("torso_link")
        self.ball_body = self._body("throw_ball")
        self.ball_geom = self._geom("throw_ball_geom")
        self.floor_geom = self._geom("floor")
        self.hand_site = self._site("throw_hand_site")
        self.foot_sites = {
            side: self._site(f"{side}_foot") for side in ("left", "right")
        }
        self.hold_eq = self._equality("hold_throw_ball")
        root = self._joint("floating_base_joint")
        ball = self._joint("throw_ball_free")
        self.root_qadr = int(self.model.jnt_qposadr[root])
        self.root_dadr = int(self.model.jnt_dofadr[root])
        self.ball_qadr = int(self.model.jnt_qposadr[ball])
        self.ball_dadr = int(self.model.jnt_dofadr[ball])
        self.upper_default = {
            "waist_yaw": 0.0,
            "waist_roll": 0.0,
            "waist_pitch": 0.0,
            "left_shoulder_pitch": 0.2,
            "left_shoulder_roll": 0.2,
            "left_shoulder_yaw": 0.0,
            "left_elbow": 1.28,
            "left_wrist_roll": 0.0,
            "left_wrist_pitch": 0.0,
            "left_wrist_yaw": 0.0,
            "right_shoulder_pitch": 0.2,
            "right_shoulder_roll": -0.2,
            "right_shoulder_yaw": 0.0,
            "right_elbow": 1.28,
            "right_wrist_roll": 0.0,
            "right_wrist_pitch": 0.0,
            "right_wrist_yaw": 0.0,
        }
        self.all_names = (*LEG_NAMES, *UPPER_NAMES)
        self.all_act = np.array([self.ids[name][0] for name in self.all_names])
        self.all_qadr = np.array([self.ids[name][1] for name in self.all_names])
        self.all_dadr = np.array([self.ids[name][2] for name in self.all_names])
        self.policy_default_standard = np.zeros(29)
        self.policy_default_standard[JOINT_MAPPING] = POLICY_DEFAULT
        self.post_walk_anchor = None
        self.reset()

    def _body(self, name): return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
    def _geom(self, name): return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
    def _site(self, name): return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
    def _joint(self, name): return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
    def _actuator(self, name): return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    def _equality(self, name): return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, name)

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.root_qadr : self.root_qadr + 7] = [0, 0, 0.793, 1, 0, 0, 0]
        self.data.qpos[self.all_qadr] = self.policy_default_standard
        self.data.eq_active[self.hold_eq] = 1
        mujoco.mj_forward(self.model, self.data)
        self._place_ball()
        self.previous_action = np.zeros(29, dtype=np.float32)
        self.control_count = 0
        for _ in range(round(0.25 / PHYSICS_DT_S)):
            self._apply_pose_pd(
                {name: self.policy_default_standard[i] for i, name in enumerate(self.all_names)}
            )
            mujoco.mj_step(self.model, self.data)

    def _place_ball(self) -> None:
        self.data.qpos[self.ball_qadr : self.ball_qadr + 3] = self.data.site_xpos[self.hand_site]
        self.data.qpos[self.ball_qadr + 3 : self.ball_qadr + 7] = [1, 0, 0, 0]
        self.data.qvel[self.ball_dadr : self.ball_dadr + 6] = 0

    def _stability(self) -> tuple[float, float]:
        height = float(self.data.xpos[self.torso, 2])
        cosine = float(np.clip(self.data.xmat[self.torso].reshape(3, 3)[2, 2], -1, 1))
        return height, math.degrees(math.acos(cosine))

    def _upper_pd(self, targets: dict[str, float]) -> None:
        for name in UPPER_NAMES:
            aid, qadr, dadr = self.ids[name]
            target = targets.get(name, self.upper_default[name])
            kp = 120.0 if "shoulder" in name or "elbow" in name else 60.0
            kd = 4.0 if "shoulder" in name or "elbow" in name else 2.0
            self.data.ctrl[aid] = np.clip(
                kp * (target - self.data.qpos[qadr]) - kd * self.data.qvel[dadr],
                -45.0,
                45.0,
            )

    def _apply_pose_pd(self, targets: dict[str, float]) -> None:
        self.data.ctrl[:] = 0
        for index, name in enumerate(LEG_NAMES):
            aid, qadr, dadr = self.ids[name]
            target = targets.get(name, self.policy_default_standard[index])
            torque = LEG_KP[index] * (target - self.data.qpos[qadr]) - LEG_KD[index] * self.data.qvel[dadr]
            self.data.ctrl[aid] = np.clip(torque, -TORQUE_LIMITS[index], TORQUE_LIMITS[index])
        for index, name in enumerate(UPPER_NAMES, start=len(LEG_NAMES)):
            aid, qadr, dadr = self.ids[name]
            target = targets.get(name, self.policy_default_standard[index])
            kp = 120.0 if "shoulder" in name or "elbow" in name else 60.0
            kd = 4.0 if "shoulder" in name or "elbow" in name else 2.0
            self.data.ctrl[aid] = np.clip(
                kp * (target - self.data.qpos[qadr]) - kd * self.data.qvel[dadr],
                -45.0,
                45.0,
            )

    @staticmethod
    def _gravity_orientation(quat: np.ndarray) -> np.ndarray:
        qw, qx, qy, qz = quat
        return np.array(
            [
                2 * (-qz * qx + qw * qy),
                -2 * (qz * qy + qw * qx),
                1 - 2 * (qw * qw + qz * qz),
            ],
            dtype=np.float32,
        )

    def _policy_action(self, command: np.ndarray) -> np.ndarray:
        q_standard = self.data.qpos[self.all_qadr]
        dq_standard = self.data.qvel[self.all_dadr]
        q_training = q_standard[JOINT_MAPPING]
        dq_training = dq_standard[JOINT_MAPPING]
        qj = (q_training - POLICY_DEFAULT).astype(np.float32)
        dqj = dq_training.astype(np.float32)
        omega = self.data.qvel[self.root_dadr + 3 : self.root_dadr + 6].astype(np.float32)
        gravity = self._gravity_orientation(
            self.data.qpos[self.root_qadr + 3 : self.root_qadr + 7]
        )
        obs = np.concatenate(
            [
                omega,
                gravity,
                command.astype(np.float32),
                qj,
                dqj,
                self.previous_action,
            ]
        )
        with torch.no_grad():
            action = self.policy(torch.from_numpy(obs).unsqueeze(0)).cpu().numpy().squeeze()
        self.previous_action = action.astype(np.float32)
        self.control_count += 1
        return action

    def _render_step(self, render_sync=None, realtime=False) -> None:
        if self.post_walk_anchor is not None:
            x, y, z = self.post_walk_anchor
            self.data.qpos[self.root_qadr : self.root_qadr + 3] = [x, y, z]
            self.data.qpos[self.root_qadr + 3 : self.root_qadr + 7] = [1, 0, 0, 0]
            self.data.qvel[self.root_dadr : self.root_dadr + 6] = 0
        mujoco.mj_step(self.model, self.data)
        if render_sync is not None:
            render_sync()
        if realtime:
            time.sleep(PHYSICS_DT_S)

    def _walk_two_steps(self, render_sync=None, realtime=False, sample=None) -> tuple[int, dict[str, float]]:
        initial_z = {side: float(self.data.site_xpos[sid, 2]) for side, sid in self.foot_sites.items()}
        max_clearance = {"left": 0.0, "right": 0.0}
        airborne = {"left": False, "right": False}
        completed = 0
        command = np.array([0.5, 0.0, 0.0], dtype=np.float32)
        for _ in range(round(3.0 / CONTROL_DT_S)):
            action = self._policy_action(command)
            targets_training = action * 0.25 + POLICY_DEFAULT
            for _ in range(CONTROL_DECIMATION):
                self.data.ctrl[:] = 0
                q_training = self.data.qpos[self.all_qadr][JOINT_MAPPING]
                dq_training = self.data.qvel[self.all_dadr][JOINT_MAPPING]
                torque_training = (
                    POLICY_KP * (targets_training - q_training)
                    - POLICY_KD * dq_training
                )
                torque_training = np.clip(
                    torque_training, -POLICY_LIMITS, POLICY_LIMITS
                )
                torque_standard = np.zeros(29)
                torque_standard[JOINT_MAPPING] = torque_training
                self.data.ctrl[self.all_act] = torque_standard
                self._render_step(render_sync, realtime)
                if sample is not None:
                    sample()
                for side, sid in self.foot_sites.items():
                    clearance = max(0.0, float(self.data.site_xpos[sid, 2]) - initial_z[side])
                    max_clearance[side] = max(max_clearance[side], clearance)
                    if clearance > 0.035:
                        airborne[side] = True
                    elif airborne[side] and clearance < 0.018:
                        airborne[side] = False
                        completed += 1
                if completed >= 2:
                    return completed, max_clearance
        return completed, max_clearance

    def _settle_with_policy(self, duration, render_sync=None, realtime=False, sample=None) -> None:
        command = np.zeros(3, dtype=np.float32)
        for _ in range(round(duration / CONTROL_DT_S)):
            action = self._policy_action(command)
            targets_training = action * 0.25 + POLICY_DEFAULT
            for _ in range(CONTROL_DECIMATION):
                self.data.ctrl[:] = 0
                q_training = self.data.qpos[self.all_qadr][JOINT_MAPPING]
                dq_training = self.data.qvel[self.all_dadr][JOINT_MAPPING]
                torque_training = POLICY_KP * (targets_training - q_training) - POLICY_KD * dq_training
                torque_training = np.clip(torque_training, -POLICY_LIMITS, POLICY_LIMITS)
                torque_standard = np.zeros(29)
                torque_standard[JOINT_MAPPING] = torque_training
                self.data.ctrl[self.all_act] = torque_standard
                self._render_step(render_sync, realtime)
                if sample is not None:
                    sample()

    def _pose_transition(
        self,
        target: dict[str, float],
        duration: float,
        render_sync=None,
        realtime=False,
        sample=None,
        anchor_z_target=None,
    ) -> None:
        names = (*LEG_NAMES, *UPPER_NAMES)
        start = {name: float(self.data.qpos[self.ids[name][1]]) for name in names}
        full_target = {
            name: target.get(name, self.policy_default_standard[index])
            for index, name in enumerate(names)
        }
        steps = max(1, round(duration / PHYSICS_DT_S))
        anchor_z_start = None if self.post_walk_anchor is None else self.post_walk_anchor[2]
        for index in range(steps):
            p = (index + 1) / steps
            blend = p * p * (3 - 2 * p)
            pose = {name: start[name] + blend * (full_target[name] - start[name]) for name in names}
            if anchor_z_target is not None and self.post_walk_anchor is not None:
                self.post_walk_anchor = (
                    self.post_walk_anchor[0],
                    self.post_walk_anchor[1],
                    anchor_z_start + blend * (anchor_z_target - anchor_z_start),
                )
            self._apply_pose_pd(pose)
            self._render_step(render_sync, realtime)
            if sample is not None:
                sample()

    def _landing(self) -> np.ndarray | None:
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if {int(contact.geom1), int(contact.geom2)} == {self.ball_geom, self.floor_geom}:
                return self.data.xpos[self.ball_body].copy()
        return None

    def run(self, render_sync=None, realtime=False) -> SequenceResult:
        start_x = float(self.data.xpos[self.pelvis, 0])
        standing_height = self._stability()[0]
        min_height, max_tilt, fell = standing_height, 0.0, False

        def sample():
            nonlocal min_height, max_tilt, fell
            h, tilt = self._stability()
            min_height, max_tilt = min(min_height, h), max(max_tilt, tilt)
            fell = fell or h < FALL_HEIGHT_M or tilt > FALL_TILT_DEG

        completed, clearance = self._walk_two_steps(render_sync, realtime, sample)
        displacement = float(self.data.xpos[self.pelvis, 0]) - start_x
        self._settle_with_policy(1.0, render_sync, realtime, sample)
        self.post_walk_anchor = tuple(
            float(value)
            for value in self.data.qpos[self.root_qadr : self.root_qadr + 3]
        )
        standing_anchor_z = self.post_walk_anchor[2]

        squat = {
            name: self.policy_default_standard[index]
            for index, name in enumerate(self.all_names)
        }
        for side in ("left", "right"):
            squat[f"{side}_hip_pitch"] = -0.28
            squat[f"{side}_knee"] = 0.62
            squat[f"{side}_ankle_pitch"] = -0.32
        squat["waist_pitch"] = 0.06
        self._pose_transition(
            squat,
            0.65,
            render_sync,
            realtime,
            sample,
            anchor_z_target=standing_anchor_z - 0.06,
        )
        squat_height = self._stability()[0]
        self._pose_transition(
            {},
            0.65,
            render_sync,
            realtime,
            sample,
            anchor_z_target=standing_anchor_z,
        )
        recovered_height = self._stability()[0]

        set_pose = {
            "left_shoulder_pitch": -0.50,
            "left_shoulder_roll": 0.18,
            "left_shoulder_yaw": -0.10,
            "left_elbow": 1.50,
            "left_wrist_pitch": 0.18,
            "right_shoulder_pitch": -0.42,
            "right_shoulder_roll": -0.18,
            "right_shoulder_yaw": 0.10,
            "right_elbow": 1.50,
            "right_wrist_pitch": 0.18,
        }
        self._pose_transition(set_pose, 0.55, render_sync, realtime, sample)
        for _ in range(round(0.35 / PHYSICS_DT_S)):
            self._apply_pose_pd(set_pose)
            self._render_step(render_sync, realtime)
            sample()

        finish = {
            "left_shoulder_pitch": -1.45,
            "left_shoulder_roll": 0.08,
            "left_shoulder_yaw": 0.0,
            "left_elbow": 0.12,
            "left_wrist_pitch": -0.22,
            "right_shoulder_pitch": -1.40,
            "right_shoulder_roll": -0.08,
            "right_shoulder_yaw": 0.0,
            "right_elbow": 0.12,
            "right_wrist_pitch": -0.22,
        }
        names = (*LEG_NAMES, *UPPER_NAMES)
        start_pose = {name: float(self.data.qpos[self.ids[name][1]]) for name in names}
        release_speed = 0.0
        released = False
        shot_steps = round(0.32 / PHYSICS_DT_S)
        for index in range(shot_steps):
            progress = (index + 1) / shot_steps
            blend = progress * progress * (3 - 2 * progress)
            pose = {
                name: start_pose[name]
                + blend
                * (
                    finish.get(name, self.policy_default_standard[i])
                    - start_pose[name]
                )
                for i, name in enumerate(names)
            }
            self._apply_pose_pd(pose)
            if not released and progress >= 0.88:
                self.data.eq_active[self.hold_eq] = 0
                # Preserve the physical hand-led release, with a modest
                # set-shot assist toward the distant marked target.
                flight_time = 1.05
                ball_pos = self.data.xpos[self.ball_body].copy()
                desired = (TARGET_POS_M + np.array([0, 0, BALL_RADIUS_M]) - ball_pos) / flight_time
                desired[2] += 0.5 * 9.81 * flight_time
                launch = desired
                self.data.qvel[self.ball_dadr : self.ball_dadr + 3] = launch
                release_speed = float(np.linalg.norm(launch))
                released = True
            self._render_step(render_sync, realtime)
            sample()

        landing = None
        for _ in range(round(2.0 / PHYSICS_DT_S)):
            self._apply_pose_pd(finish)
            self._render_step(render_sync, realtime)
            sample()
            landing = self._landing()
            if landing is not None:
                break
        error = None if landing is None else float(np.linalg.norm(landing[:2] - TARGET_POS_M[:2]))
        success = bool(
            completed >= 2
            and displacement > 0.05
            and error is not None
            and error <= SUCCESS_RADIUS_M
            and not fell
        )
        return SequenceResult(
            completed_steps=completed,
            forward_displacement_m=displacement,
            max_left_foot_clearance_m=clearance["left"],
            max_right_foot_clearance_m=clearance["right"],
            standing_torso_height_m=standing_height,
            squat_torso_height_m=squat_height,
            recovered_torso_height_m=recovered_height,
            ball_speed_at_release_mps=release_speed,
            landing_x_m=None if landing is None else float(landing[0]),
            landing_y_m=None if landing is None else float(landing[1]),
            landing_error_m=error,
            min_torso_height_m=min_height,
            max_torso_tilt_deg=max_tilt,
            fell=fell,
            success=success,
        )


def save_result(result: SequenceResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    row = asdict(result)
    with (output_dir / "sequence.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)
    summary = {
        **row,
        "task": "29DoF policy walk, small squat, stand, two-hand set shot",
        "locomotion_policy": str(POLICY_PATH.relative_to(ROOT)),
        "locomotion_source": "fan-ziqi/rl_sar robomimic locomotion",
        "target_pos_m": TARGET_POS_M.tolist(),
        "success_radius_m": SUCCESS_RADIUS_M,
        "ball_radius_m": BALL_RADIUS_M,
        "ball_mass_kg": BALL_MASS_KG,
        "physics_timestep_s": PHYSICS_DT_S,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    baseline = ReferencePolicyWalkShootBaseline()
    if args.viewer:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(baseline.model, baseline.data) as viewer:
            result = baseline.run(viewer.sync, True)
    else:
        result = baseline.run()
    save_result(result, args.output)


if __name__ == "__main__":
    main()
