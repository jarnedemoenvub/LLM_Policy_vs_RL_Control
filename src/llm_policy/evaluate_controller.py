import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from src.envs.make_env import make_env
from src.llm_policy.controller_template import controller, random_controller


def evaluate_controller(
    controller_fn,
    controller_name,
    env_id="Pendulum-v1",
    episodes=100,
    seed=42,
):
    """
    Evaluate a controller function on a Gymnasium environment.

    The controller must take an observation as input and return an action.
    """
    episode_rewards = []

    for episode in range(episodes):
        env = make_env(env_id)
        obs, info = env.reset(seed=seed + episode)

        done = False
        total_reward = 0.0

        while not done:
            action = controller_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            done = terminated or truncated

        episode_rewards.append(float(total_reward))
        env.close()

    summary = {
        "controller_name": controller_name,
        "env_id": env_id,
        "episodes": episodes,
        "seed": seed,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
    }

    episode_data = [
        {
            "controller_name": controller_name,
            "env_id": env_id,
            "seed": seed,
            "episode": i + 1,
            "episode_seed": seed + i,
            "reward": reward,
        }
        for i, reward in enumerate(episode_rewards)
    ]

    return summary, episode_data


def save_controller_results(summary, episode_data, output_dir="results/controllers"):
    """
    Save one controller's summary and episode-level rewards separately.
    """
    os.makedirs(output_dir, exist_ok=True)

    controller_name = summary["controller_name"]

    summary_path = os.path.join(output_dir, f"{controller_name}_summary.csv")
    episodes_path = os.path.join(output_dir, f"{controller_name}_episode_rewards.csv")

    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    pd.DataFrame(episode_data).to_csv(episodes_path, index=False)

    print(f"Saved summary to: {summary_path}")
    print(f"Saved episode rewards to: {episodes_path}")


if __name__ == "__main__":
    output_dir = "results/controllers"

    controllers = {
        "manual_controller": controller,
        "random_controller": random_controller,
    }

    all_summaries = []

    for controller_name, controller_fn in controllers.items():
        summary, episode_data = evaluate_controller(
            controller_fn=controller_fn,
            controller_name=controller_name,
            env_id="Pendulum-v1",
            episodes=100,
            seed=42,
        )

        save_controller_results(
            summary=summary,
            episode_data=episode_data,
            output_dir=output_dir,
        )

        all_summaries.append(summary)

    all_summaries_path = os.path.join(output_dir, "all_controller_summaries.csv")
    pd.DataFrame(all_summaries).to_csv(all_summaries_path, index=False)

    print("\nAll controller summaries")
    print("------------------------")
    print(pd.DataFrame(all_summaries))
    print(f"\nSaved combined summary to: {all_summaries_path}")