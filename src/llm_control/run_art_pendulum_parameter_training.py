from __future__ import annotations

import asyncio
import json
import math
import re
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd

import art
from art.local import LocalBackend
from art.utils.litellm import convert_litellm_choice_to_openai
from litellm import acompletion


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "results" / "art_pendulum"
TRAJECTORIES_DIR = RESULTS_DIR / "trajectories"
POLICIES_DIR = RESULTS_DIR / "policies"
EVAL_LOGS_DIR = RESULTS_DIR / "eval_logs"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TRAJECTORIES_DIR.mkdir(parents=True, exist_ok=True)
POLICIES_DIR.mkdir(parents=True, exist_ok=True)
EVAL_LOGS_DIR.mkdir(parents=True, exist_ok=True)


PROJECT_NAME = "llm-policy-vs-rl-control"
MODEL_NAME = "art-pendulum-kp-kd-001"

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

MAX_SEQ_LENGTH = 512
GPU_MEMORY_UTILIZATION = 0.50

# Small feasibility run.
N_TRAINING_STEPS = 1
GROUPS_PER_STEP = 1
ROLLOUTS_PER_GROUP = 1

# Evaluation budget per rollout.
ENV_ID = "Pendulum-v1"
N_EVAL_EPISODES_PER_ROLLOUT = 1
SEED = 42

# Final evaluation after ART.
N_FINAL_EVAL_EPISODES = 5


SYSTEM_PROMPT = """
You generate controller parameters for Gymnasium Pendulum-v1.

The environment observation is:
obs[0] = cos(theta)
obs[1] = sin(theta)
obs[2] = theta_dot

The action is one continuous torque in the range [-2, 2].

Your task is to output two floating point numbers:
Kp and Kd

They will be used in this controller:

theta = atan2(sin_theta, cos_theta)
torque = Kp * theta + Kd * theta_dot
torque = clipped to [-2, 2]

Higher reward is better. Pendulum rewards are negative, so values closer to zero are better.

Return only valid JSON with this exact format:
{"Kp": 1.0, "Kd": 0.5}

Do not include explanations.
"""


USER_PROMPT = """
Generate Kp and Kd for a Pendulum-v1 controller.
The controller should swing up and stabilize the pendulum.
Return only JSON.
"""


def extract_json_object(text: str) -> dict[str, float]:
    text = text.strip()

    # First try direct JSON.
    try:
        data = json.loads(text)
        return {
            "Kp": float(data["Kp"]),
            "Kd": float(data["Kd"]),
        }
    except Exception:
        pass

    # Fallback: find first JSON-looking object.
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if match:
        data = json.loads(match.group(0))
        return {
            "Kp": float(data["Kp"]),
            "Kd": float(data["Kd"]),
        }

    raise ValueError(f"Could not parse Kp/Kd JSON from model output: {text}")


def clip_parameter(value: float, low: float = -10.0, high: float = 10.0) -> float:
    return float(max(low, min(high, value)))


def pendulum_policy(obs: np.ndarray, kp: float, kd: float) -> np.ndarray:
    cos_theta = float(obs[0])
    sin_theta = float(obs[1])
    theta_dot = float(obs[2])

    theta = math.atan2(sin_theta, cos_theta)

    torque = kp * theta + kd * theta_dot
    torque = max(-2.0, min(2.0, torque))

    return np.array([torque], dtype=np.float32)


def evaluate_kp_kd(
    kp: float,
    kd: float,
    n_episodes: int,
    seed: int,
    trace_path: Path | None = None,
) -> dict[str, Any]:
    env = gym.make(ENV_ID)

    episode_rewards = []
    total_steps = 0

    trace_file = None
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_file = trace_path.open("w", encoding="utf-8")

    try:
        for episode in range(n_episodes):
            obs, info = env.reset(seed=seed + episode)

            done = False
            truncated = False
            ep_reward = 0.0
            step = 0

            while not (done or truncated):
                action = pendulum_policy(obs, kp=kp, kd=kd)
                next_obs, reward, done, truncated, info = env.step(action)

                if trace_file is not None:
                    trace_file.write(
                        json.dumps(
                            {
                                "episode": episode,
                                "step": step,
                                "observation": obs.astype(float).tolist(),
                                "action": action.astype(float).tolist(),
                                "reward": float(reward),
                                "next_observation": next_obs.astype(float).tolist(),
                                "done": bool(done),
                                "truncated": bool(truncated),
                            }
                        )
                        + "\n"
                    )

                ep_reward += float(reward)
                obs = next_obs
                step += 1
                total_steps += 1

            episode_rewards.append(ep_reward)

    finally:
        if trace_file is not None:
            trace_file.close()
        env.close()

    rewards = np.array(episode_rewards, dtype=np.float64)

    return {
        "kp": kp,
        "kd": kd,
        "n_episodes": n_episodes,
        "seed": seed,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "episode_rewards": episode_rewards,
        "total_env_steps": total_steps,
    }


def normalize_reward(mean_reward: float) -> float:

    return float(max(0.0, min(1.0, (mean_reward + 1600.0) / 1600.0)))


def save_policy_params(path: Path, kp: float, kd: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "Kp": kp,
                "Kd": kd,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

async def rollout(model: art.Model, step: int, group_id: int, rollout_id: int) -> art.Trajectory:
 
    traj = art.Trajectory(
        reward=0.0,
        messages_and_choices=[],
        metadata={
            "step": step,
            "group_id": group_id,
            "rollout_id": rollout_id,
        },
    )

    traj.messages_and_choices = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ]

    if model.trainable:
        litellm_model_name = f"hosted_vllm/{model.name}"
    else:
        litellm_model_name = model.name

    response = await acompletion(
        model=litellm_model_name,
        base_url=model.inference_base_url,
        api_key=model.inference_api_key,
        temperature=0.7,
        messages=traj.messages(),
        caching=False,
    )

    assistant_choice = convert_litellm_choice_to_openai(response.choices[0])
    traj.messages_and_choices.append(assistant_choice)

    assistant_text = response.choices[0].message.content

    try:
        params = extract_json_object(assistant_text)
        kp = clip_parameter(params["Kp"])
        kd = clip_parameter(params["Kd"])

        trace_path = (
            TRAJECTORIES_DIR
            / f"step{step:03d}_group{group_id:03d}_rollout{rollout_id:03d}_trace.jsonl"
        )

        eval_result = evaluate_kp_kd(
            kp=kp,
            kd=kd,
            n_episodes=N_EVAL_EPISODES_PER_ROLLOUT,
            seed=SEED + step * 1000 + group_id * 100 + rollout_id,
            trace_path=trace_path,
        )

        art_reward = normalize_reward(eval_result["mean_reward"])

        traj.reward = art_reward
        traj.metadata.update(
            {
                "valid": True,
                "kp": kp,
                "kd": kd,
                "mean_reward": eval_result["mean_reward"],
                "std_reward": eval_result["std_reward"],
                "min_reward": eval_result["min_reward"],
                "max_reward": eval_result["max_reward"],
                "art_reward": art_reward,
                "trace_path": str(trace_path),
            }
        )

    except Exception as exc:
        # Invalid JSON or bad generation gets zero reward.
        traj.reward = 0.0
        traj.metadata.update(
            {
                "valid": False,
                "error": repr(exc),
                "assistant_text": assistant_text,
                "art_reward": 0.0,
            }
        )

    return traj

async def main() -> None:
    print("=" * 80)
    print("Starting ART Pendulum parameter-training experiment")
    print("=" * 80)

    model = art.TrainableModel(
        name=MODEL_NAME,
        project=PROJECT_NAME,
        base_model=BASE_MODEL,
    )

    # Config based on the AutoRL local example style.
    model._internal_config = art.dev.InternalModelConfig(
        init_args=art.dev.InitArgs(
            max_seq_length=MAX_SEQ_LENGTH,
        ),
        engine_args=art.dev.EngineArgs(
            enforce_eager=True,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        ),
    )

    backend = LocalBackend(
        in_process=True,
        path=str(PROJECT_ROOT / ".art"),
    )

    await model.register(backend)

    all_rows = []

    for step in range(N_TRAINING_STEPS):
        print("\n" + "-" * 80)
        print(f"ART training step {step + 1}/{N_TRAINING_STEPS}")
        print("-" * 80)

        groups = []

        for group_id in range(GROUPS_PER_STEP):
            group = art.TrajectoryGroup(
                rollout(model, step=step, group_id=group_id, rollout_id=rollout_id)
                for rollout_id in range(ROLLOUTS_PER_GROUP)
            )
            groups.append(group)

        finished_groups = await art.gather_trajectory_groups(
            groups,
            pbar_desc="Generating Pendulum parameter rollouts",
            max_exceptions=GROUPS_PER_STEP * ROLLOUTS_PER_GROUP,
        )

        # Save rollout metadata.
        for group_id, group in enumerate(finished_groups):
            for rollout_id, traj in enumerate(group.trajectories):
                row = {
                    "step": step,
                    "group_id": group_id,
                    "rollout_id": rollout_id,
                    "reward_for_art": traj.reward,
                    **traj.metadata,
                }
                all_rows.append(row)

        rollouts_df = pd.DataFrame(all_rows)
        rollouts_path = RESULTS_DIR / "art_rollouts.csv"
        rollouts_df.to_csv(rollouts_path, index=False)

        print("Rollout rewards:")
        print(
            rollouts_df[rollouts_df["step"] == step][
                [
                    "step",
                    "group_id",
                    "rollout_id",
                    "valid",
                    "kp",
                    "kd",
                    "mean_reward",
                    "reward_for_art",
                ]
            ].to_string(index=False)
        )

        # Train ART model on objective rewards.
        await model.delete_checkpoints()
        await model.train(
            finished_groups,
            config=art.TrainConfig(
                learning_rate=1e-5,
            ),
            _config={
                "logprob_calculation_chunk_size": 8,
            },
        )

        print(f"Completed ART training step {step}")

    print("\nART training completed.")

    rollouts_df = pd.DataFrame(all_rows)
    valid_df = rollouts_df[rollouts_df["valid"] == True].copy()

    if valid_df.empty:
        print("No valid ART rollouts were produced.")
        return

    best = valid_df.sort_values("mean_reward", ascending=False).iloc[0]

    best_kp = float(best["kp"])
    best_kd = float(best["kd"])

    best_params_path = POLICIES_DIR / "best_art_kp_kd.json"
    save_policy_params(best_params_path, best_kp, best_kd)

    final_trace_path = EVAL_LOGS_DIR / "best_art_100ep_trace.jsonl"
    final_eval = evaluate_kp_kd(
        kp=best_kp,
        kd=best_kd,
        n_episodes=N_FINAL_EVAL_EPISODES,
        seed=SEED,
        trace_path=final_trace_path,
    )

    final_eval["best_params_path"] = str(best_params_path)
    final_eval["final_trace_path"] = str(final_trace_path)

    final_eval_path = EVAL_LOGS_DIR / "best_art_100ep_eval.json"
    final_eval_path.write_text(json.dumps(final_eval, indent=2), encoding="utf-8")

    final_summary_path = RESULTS_DIR / "art_final_summary.csv"
    pd.DataFrame(
        [
            {
                "method": "ART_parameter_training",
                "base_model": BASE_MODEL,
                "kp": best_kp,
                "kd": best_kd,
                "mean_reward_100ep": final_eval["mean_reward"],
                "std_reward_100ep": final_eval["std_reward"],
                "min_reward_100ep": final_eval["min_reward"],
                "max_reward_100ep": final_eval["max_reward"],
                "n_eval_episodes": N_FINAL_EVAL_EPISODES,
                "total_env_steps": final_eval["total_env_steps"],
                "params_path": str(best_params_path),
                "eval_json_path": str(final_eval_path),
                "trace_path": str(final_trace_path),
            }
        ]
    ).to_csv(final_summary_path, index=False)

    print("\nBest ART rollout:")
    print(f"Kp: {best_kp:.4f}")
    print(f"Kd: {best_kd:.4f}")
    print(f"Training mean reward: {float(best['mean_reward']):.3f}")

    print("\nFinal 100-episode ART evaluation:")
    print(f"Mean reward: {final_eval['mean_reward']:.3f}")
    print(f"Std reward: {final_eval['std_reward']:.3f}")
    print(f"Min reward: {final_eval['min_reward']:.3f}")
    print(f"Max reward: {final_eval['max_reward']:.3f}")

    print("\nSaved:")
    print(f"- {rollouts_path}")
    print(f"- {best_params_path}")
    print(f"- {final_eval_path}")
    print(f"- {final_summary_path}")


if __name__ == "__main__":
    asyncio.run(main())