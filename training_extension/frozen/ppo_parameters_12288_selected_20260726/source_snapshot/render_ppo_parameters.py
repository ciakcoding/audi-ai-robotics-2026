from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from .optimize_direct import controller_action
from .sac_parameter_env import SACShotParameterEnv


def annotate(frame, text, color=(255, 255, 255)):
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((18, 16, 1210, 62), fill=(0, 0, 0))
    draw.text((32, 28), text, fill=color)
    return np.asarray(image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vecnormalize", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=100_000)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=60)
    args = parser.parse_args()

    shot_env = SACShotParameterEnv()
    vector_env = DummyVecEnv([lambda: shot_env])
    vector_env = VecNormalize.load(args.vecnormalize, vector_env)
    vector_env.training = False
    vector_env.norm_reward = False
    model = PPO.load(args.model, env=vector_env, device="cpu")
    vector_env.seed(args.seed)
    observation = vector_env.reset()
    residual, _ = model.predict(observation, deterministic=True)
    parameters = (
        shot_env.expert_parameters
        + shot_env.parameter_scales * residual[0]
    )
    env = shot_env.base

    renderer = mujoco.Renderer(
        env.model, height=args.height, width=args.width
    )
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.9, 0.0, 1.0]
    camera.distance = 3.2
    camera.azimuth = 145
    camera.elevation = -12

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        args.output, fps=args.fps, codec="libx264", quality=8
    )
    terminated = truncated = False
    info = {}
    frame_stride = max(1, int(round(1.0 / (args.fps * 0.002))))
    physical_step = 0
    while not (terminated or truncated):
        action = controller_action(env, parameters)
        _, _, terminated, truncated, info = env.step(action)
        physical_step += env.control_substeps
        if physical_step % frame_stride < env.control_substeps:
            renderer.update_scene(env.data, camera=camera)
            status = (
                "RL=PPO-PARAMETER  "
                f"DIRECT={'YES' if info['direct_shot'] else 'PENDING'}  "
                f"flight={info['airborne_horizontal_distance']:.2f} m  "
                f"backboard={'YES' if info['touched_backboard'] else 'NO'}  "
                f"fall={'YES' if info['has_fallen'] else 'NO'}"
            )
            writer.append_data(annotate(renderer.render(), status))
    for _ in range(args.fps):
        renderer.update_scene(env.data, camera=camera)
        status = (
            "RL=PPO-PARAMETER  "
            f"DIRECT={'YES' if info['direct_shot'] else 'NO'}  "
            f"crossing_error={info['crossing_xy_error'] * 100:.2f} cm  "
            f"flight={info['airborne_horizontal_distance']:.2f} m  "
            f"backboard={'YES' if info['touched_backboard'] else 'NO'}  "
            f"fall={'YES' if info['has_fallen'] else 'NO'}"
        )
        writer.append_data(
            annotate(renderer.render(), status, (100, 255, 100))
        )
    writer.close()
    renderer.close()
    vector_env.close()
    print(json.dumps(info, indent=2))
    print(args.output.resolve())


if __name__ == "__main__":
    main()
