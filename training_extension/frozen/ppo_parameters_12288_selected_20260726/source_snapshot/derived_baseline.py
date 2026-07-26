"""Training baseline derived from the teammate's unchanged LEVEL03 baseline.

Only the visibly problematic right-foot landing and upper-body throw keyframes
are adjusted.  The target, physics, scoring, walk structure, dip, and leg drive
remain unchanged.  This reference is intentionally imperfect so learning still
has a measurable throwing task.
"""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
# Keep the teammate's current baseline untouched and derive from it.  The
# remote feature/simulation03 branch moved v030 into "old scripts" and made
# v031 the supported LEVEL03 entry point.
ORIGINAL = ROOT / "scripts" / "view_baselines_LEVEL03_v031!.py"


def _load_original():
    spec = importlib.util.spec_from_file_location("peer_level03_original", ORIGINAL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load original baseline: {ORIGINAL}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.G1FixedBodyThrowEnv, module.OptionDBasketballPolicy


PeerEnv, OriginalPolicy = _load_original()


class TrainingBasketballPolicy(OriginalPolicy):
    """Conservative, human-readable starting point for subsequent learning."""

    def __init__(self, env):
        super().__init__(env)
        self.keyframes = copy.deepcopy(self.keyframes)

        # Reduce the exaggerated right-leg swing and keep the knee flexed as
        # the foot returns to the floor.
        self.keyframes[100].update(
            {
                "right_hip_pitch_joint": -0.72,
                "right_knee_joint": 1.05,
                "right_ankle_pitch_joint": -0.30,
            }
        )
        self.keyframes[150].update(
            {
                "right_hip_pitch_joint": -0.32,
                "right_knee_joint": 0.38,
                "right_ankle_pitch_joint": -0.16,
                "left_hip_pitch_joint": 0.28,
                "left_knee_joint": 0.36,
                "left_ankle_pitch_joint": 0.02,
            }
        )
        self.keyframes[200].update(
            {
                "right_hip_pitch_joint": 0.02,
                "right_knee_joint": 0.34,
                "right_ankle_pitch_joint": -0.15,
            }
        )
        # Mirror the same controlled landing on the second (left-foot) step.
        # The original nearly locked the knee while the ankle was still
        # heel-down, producing a conspicuous heel strike.
        self.keyframes[250].update(
            {
                "left_hip_pitch_joint": -0.72,
                "left_knee_joint": 1.05,
                "left_ankle_pitch_joint": 0.00,
            }
        )
        self.keyframes[300].update(
            {
                "left_hip_pitch_joint": -0.32,
                "left_knee_joint": 0.38,
                "left_ankle_pitch_joint": 0.12,
                "right_hip_pitch_joint": 0.28,
                "right_knee_joint": 0.36,
                "right_ankle_pitch_joint": 0.02,
            }
        )

        # Rough two-hand set shot. It is deliberately not ballistically tuned
        # so later trajectory optimization has a measurable task.
        self.keyframes[380].update(
            {
                "right_shoulder_pitch_joint": -1.05,
                "right_shoulder_roll_joint": 0.10,
                "right_elbow_joint": 1.55,
                "right_wrist_pitch_joint": -0.65,
                "left_shoulder_pitch_joint": -1.00,
                "left_shoulder_roll_joint": -0.10,
                "left_elbow_joint": 1.50,
                "left_wrist_pitch_joint": -0.45,
            }
        )
        self.keyframes[400].update(
            {
                "right_shoulder_pitch_joint": -1.35,
                "right_shoulder_roll_joint": 0.08,
                "right_elbow_joint": 0.85,
                "right_wrist_pitch_joint": -0.20,
                "left_shoulder_pitch_joint": -1.28,
                "left_shoulder_roll_joint": -0.08,
                "left_elbow_joint": 0.95,
                "left_wrist_pitch_joint": -0.15,
            }
        )
        self.keyframes[410].update(
            {
                "right_shoulder_pitch_joint": -1.60,
                "right_shoulder_roll_joint": 0.08,
                "right_elbow_joint": 0.38,
                "right_wrist_pitch_joint": 0.35,
                # Guide hand begins peeling away just before release so its
                # forearm never has to pass through the shooting arm.
                "left_shoulder_pitch_joint": -0.78,
                "left_shoulder_roll_joint": -0.30,
                "left_elbow_joint": 0.82,
                "left_wrist_pitch_joint": 0.04,
            }
        )
        # Immediate post-release split: the right shooting hand keeps its
        # follow-through while the left guide hand exits down and outward.
        # The early 412 frame prevents the forearms from swapping sides before
        # reaching the more relaxed 420/440 follow-through.
        self.keyframes[412] = copy.deepcopy(self.keyframes[410])
        self.keyframes[412].update(
            {
                "right_shoulder_pitch_joint": -1.58,
                "right_shoulder_roll_joint": 0.20,
                "right_elbow_joint": 0.42,
                "right_wrist_pitch_joint": 0.22,
                "left_shoulder_pitch_joint": -0.20,
                "left_shoulder_roll_joint": -0.52,
                "left_elbow_joint": 0.72,
                "left_wrist_pitch_joint": 0.00,
            }
        )
        self.keyframes[420] = copy.deepcopy(self.keyframes[410])
        self.keyframes[420].update(
            {
                "right_shoulder_pitch_joint": -1.58,
                "right_shoulder_roll_joint": 0.20,
                "right_elbow_joint": 0.42,
                "right_wrist_pitch_joint": 0.22,
                "left_shoulder_pitch_joint": -0.28,
                "left_shoulder_roll_joint": -0.48,
                "left_elbow_joint": 0.78,
                "left_wrist_pitch_joint": 0.02,
            }
        )
        self.keyframes[440].update(
            {
                "right_shoulder_pitch_joint": -1.62,
                "right_shoulder_roll_joint": 0.18,
                "right_elbow_joint": 0.42,
                "right_wrist_pitch_joint": 0.22,
                "left_shoulder_pitch_joint": -0.32,
                "left_shoulder_roll_joint": -0.38,
                "left_elbow_joint": 0.82,
                "left_wrist_pitch_joint": 0.02,
            }
        )
        # Recover along two separated lanes instead of sweeping both forearms
        # through the same centreline.
        self.keyframes[500].update(
            {
                "right_shoulder_roll_joint": 0.12,
                "left_shoulder_roll_joint": -0.12,
                "right_wrist_pitch_joint": 0.05,
                "left_wrist_pitch_joint": 0.05,
            }
        )
        self.frame_times = sorted(self.keyframes)
