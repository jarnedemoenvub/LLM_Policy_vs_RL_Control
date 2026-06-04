import os
import sys
import time
import argparse
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from stable_baselines3 import PPO

from src.envs.make_env import make_env
from src.llm_policy.controller_template import controller as manual_controller


def random_controller(obs):
    """
    Random controller for Pendulum-v1.
    Samples torque uniformly from [-2, 2].
    """
    return np.array([np.random.uniform(-2.0, 2.0)], dtype=np.float32)


def watch_function_controller(
    controller_fn,
    env_id="Pendulum-v1",
    episodes=5,
    seed=42,
    sleep_time=0.02,
):
    env = make_env(env_id, render_mode="human")

    for episode in range(episodes):
        obs, info = env.reset(seed=seed + episode)
        done = False
        total_reward = 0.0

        while not done:
            action = controller_fn(obs)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            done = terminated or truncated
            time.sleep(sleep_time)

        print(f"Episode {episode + 1}: reward = {total_reward:.2f}")

    env.close()


def watch_ppo(
    model_path,
    env_id="Pendulum-v1",
    episodes=5,
    seed=42,
    sleep_time=0.02,
):
    env = make_env(env_id, render_mode="human")
    model = PPO.load(model_path)

    for episode in range(episodes):
        obs, info = env.reset(seed=seed + episode)
        done = False
        total_reward = 0.0

        while not done:
            action, _states = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            done = terminated or truncated
            time.sleep(sleep_time)

        print(f"Episode {episode + 1}: reward = {total_reward:.2f}")

    env.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--method",
        choices=["random", "manual", "ppo"],
        required=True,
        help="Which policy to animate.",
    )

    parser.add_argument(
        "--model-path",
        default="results/raw/ppo_Pendulum-v1_seed42_300000.zip",
        help="Path to PPO model. Only used when --method ppo.",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes to render.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=0.02,
        help="Delay between frames.",
    )

    args = parser.parse_args()

    if args.method == "random":
        print("Watching random policy")
        watch_function_controller(
            random_controller,
            episodes=args.episodes,
            seed=args.seed,
            sleep_time=args.sleep,
        )

    elif args.method == "ppo":
        print(f"Watching PPO policy from: {args.model_path}")
        watch_ppo(
            model_path=args.model_path,
            episodes=args.episodes,
            seed=args.seed,
            sleep_time=args.sleep,
        )


if __name__ == "__main__":
    main()