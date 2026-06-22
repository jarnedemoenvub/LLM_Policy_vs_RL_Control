from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from src.llm_control.safety import call_policy_with_timeout


def evaluate_generated_policy(
    code: str,
    env_id: str,
    n_episodes: int,
    seed: int,
    trace_path: Path,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """
    Evaluates a generated policy in a Gymnasium environment.
    Stores full sensory-motor trace as JSONL.
    """
    env = gym.make(env_id)

    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    failures: list[str] = []

    total_env_steps = 0

    trace_path.parent.mkdir(parents=True, exist_ok=True)

    with trace_path.open("w", encoding="utf-8") as trace_file:
        for episode in range(n_episodes):
            obs, info = env.reset(seed=seed + episode)

            done = False
            truncated = False
            episode_reward = 0.0
            step = 0

            while not (done or truncated):
                try:
                    action = call_policy_with_timeout(code, obs)
                except Exception as exc:
                    failures.append(
                        f"episode={episode}, step={step}, error={repr(exc)}"
                    )
                    # If policy fails, use zero torque and continue logging failure.
                    action = np.array([0.0], dtype=np.float32)

                next_obs, reward, done, truncated, info = env.step(action)

                record = {
                    "episode": episode,
                    "step": step,
                    "observation": obs.astype(float).tolist(),
                    "action": action.astype(float).tolist(),
                    "reward": float(reward),
                    "next_observation": next_obs.astype(float).tolist(),
                    "done": bool(done),
                    "truncated": bool(truncated),
                }
                trace_file.write(json.dumps(record) + "\n")

                episode_reward += float(reward)
                obs = next_obs
                step += 1
                total_env_steps += 1

                if max_steps is not None and step >= max_steps:
                    break

            episode_rewards.append(episode_reward)
            episode_lengths.append(step)

    env.close()

    rewards_array = np.array(episode_rewards, dtype=np.float64)

    return {
        "env_id": env_id,
        "n_episodes": n_episodes,
        "seed": seed,
        "mean_reward": float(np.mean(rewards_array)),
        "std_reward": float(np.std(rewards_array)),
        "min_reward": float(np.min(rewards_array)),
        "max_reward": float(np.max(rewards_array)),
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "total_env_steps": total_env_steps,
        "n_failures": len(failures),
        "failures": failures[:20],
    }


def load_last_n_trace_steps(trace_path: Path, n: int = 20) -> list[dict[str, Any]]:
    """
    Loads the last n sensory-motor records from a JSONL trace.
    Useful later for iterative refinement.
    """
    with trace_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    selected = lines[-n:]
    return [json.loads(line) for line in selected]