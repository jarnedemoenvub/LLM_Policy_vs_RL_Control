# src/llm_control/run_iterative_refinement_pendulum.py

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.llm_control.code_extractor import extract_python_code
from src.llm_control.config import (
    ENV_ID,
    MAX_RESPONSE_TOKENS,
    SEED,
    MODELS
)
from src.llm_control.evaluate_policy import (
    evaluate_generated_policy,
    load_last_n_trace_steps,
)
from src.llm_control.ollama_client import query_ollama
from src.llm_control.prompts import prompt_4_refine_policy

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ONE_SHOT_DIR = PROJECT_ROOT / "results" / "llm_one_shot" / "pendulum"
ONE_SHOT_GLOBAL_SUMMARY = ONE_SHOT_DIR / "summary.csv"

REFINE_BASE_DIR = PROJECT_ROOT / "results" / "llm_refinement" / "pendulum"
GLOBAL_REFINEMENT_SUMMARY = REFINE_BASE_DIR / "refinement_summary_all_models.csv"

N_REFINEMENT_ITERATIONS = 10
N_EVAL_EPISODES = 10
LAST_N_TRACE_STEPS = 20

REFINEMENT_TEMPERATURE = 0.4


def make_safe_model_name(model: str) -> str:
    return model.replace(":", "_").replace("/", "_").replace(".", "_")


def get_refine_dirs(model: str) -> dict[str, Path]:
    model_safe = make_safe_model_name(model)
    refine_dir = REFINE_BASE_DIR / model_safe

    return {
        "refine_dir": refine_dir,
        "prompts_dir": refine_dir / "prompts",
        "policies_dir": refine_dir / "policies",
        "eval_logs_dir": refine_dir / "eval_logs",
        "traces_dir": refine_dir / "sensory_motor_traces",
        "summary_path": refine_dir / "refinement_summary.csv",
    }


def get_one_shot_model_summary_path(model: str) -> Path:
    model_safe = make_safe_model_name(model)
    return ONE_SHOT_DIR / model_safe / "summary.csv"


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def append_csv(row: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        df = pd.read_csv(path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(path, index=False)


def append_refinement_summary(row: dict[str, Any], model: str) -> None:
    dirs = get_refine_dirs(model)

    append_csv(row, dirs["summary_path"])
    append_csv(row, GLOBAL_REFINEMENT_SUMMARY)


def format_sensory_motor_steps(steps: list[dict[str, Any]]) -> str:
    lines = []

    for item in steps:
        obs = item["observation"]
        action = item["action"]
        reward = item["reward"]

        cos_theta = obs[0]
        sin_theta = obs[1]
        theta_dot = obs[2]
        torque = action[0]

        lines.append(
            f"step={item['step']}: "
            f"obs=[cos(theta)={cos_theta:.3f}, "
            f"sin(theta)={sin_theta:.3f}, "
            f"theta_dot={theta_dot:.3f}], "
            f"action=[torque={torque:.3f}], "
            f"reward={reward:.3f}"
        )

    return "\n".join(lines)


def select_best_one_shot_policy(model: str) -> dict[str, Any]:
    model_summary_path = get_one_shot_model_summary_path(model)

    if model_summary_path.exists():
        df = pd.read_csv(model_summary_path)
        summary_path_used = model_summary_path
    elif ONE_SHOT_GLOBAL_SUMMARY.exists():
        df = pd.read_csv(ONE_SHOT_GLOBAL_SUMMARY)
        df = df[df["model"] == model].copy()
        summary_path_used = ONE_SHOT_GLOBAL_SUMMARY
    else:
        raise FileNotFoundError(
            "Could not find one-shot summary. Expected either:\n"
            f"{model_summary_path}\n"
            f"{ONE_SHOT_GLOBAL_SUMMARY}"
        )

    df_valid = df[df["status"] == "valid"].copy()

    if df_valid.empty:
        raise RuntimeError(
            f"No valid one-shot runs found for model {model} "
            f"in {summary_path_used}"
        )

    df_valid = df_valid.sort_values("mean_reward", ascending=False)
    best_row = df_valid.iloc[0].to_dict()

    print("Selected best one-shot policy:")
    print(f"Model: {model}")
    print(f"Run ID: {best_row['run_id']}")
    print(f"Mean reward: {float(best_row['mean_reward']):.3f}")
    print(f"Policy path: {best_row['policy_path']}")

    return best_row


def run_refinement_for_model(model: str) -> None:
    model_safe = make_safe_model_name(model)
    dirs = get_refine_dirs(model)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    experiment_id = f"refine_{ENV_ID}_{model_safe}_{timestamp}"

    print("\n" + "#" * 80)
    print(f"STARTING REFINEMENT FOR MODEL: {model}")
    print(f"Experiment ID: {experiment_id}")
    print("#" * 80)

    try:
        best_one_shot = select_best_one_shot_policy(model)
    except Exception as exc:
        print(f"Skipping model {model}: {repr(exc)}")
        return

    current_policy_path = Path(best_one_shot["policy_path"])
    current_trace_path = Path(best_one_shot["trace_path"])

    current_policy_code = load_text(current_policy_path)
    current_mean_reward = float(best_one_shot["mean_reward"])
    current_std_reward = float(best_one_shot["std_reward"])

    best_policy_code = current_policy_code
    best_mean_reward = current_mean_reward
    best_std_reward = current_std_reward
    best_iteration = 0

    previous_policy_code = None
    previous_mean_reward = None

    initial_policy_path = dirs["policies_dir"] / f"{experiment_id}_iter000_initial_policy.py"
    save_text(initial_policy_path, current_policy_code)

    initial_row = {
        "experiment_id": experiment_id,
        "iteration": 0,
        "source": "best_one_shot",
        "model": model,
        "model_safe": model_safe,
        "temperature": REFINEMENT_TEMPERATURE,
        "env_id": ENV_ID,
        "mean_reward": current_mean_reward,
        "std_reward": current_std_reward,
        "min_reward": None,
        "max_reward": None,
        "total_env_steps": 0,
        "n_failures": 0,
        "best_mean_reward_so_far": best_mean_reward,
        "best_iteration_so_far": best_iteration,
        "is_best": True,
        "status": "valid",
        "policy_path": str(initial_policy_path),
        "trace_path": str(current_trace_path),
        "prompt_path": "",
        "eval_path": "",
    }

    append_refinement_summary(initial_row, model)

    for iteration in range(1, N_REFINEMENT_ITERATIONS + 1):
        print("\n" + "-" * 80)
        print(f"Model: {model}")
        print(f"Refinement iteration {iteration}/{N_REFINEMENT_ITERATIONS}")
        print("-" * 80)

        try:
            last_steps = load_last_n_trace_steps(current_trace_path, n=LAST_N_TRACE_STEPS)
            sensory_motor_summary = format_sensory_motor_steps(last_steps)
        except Exception as exc:
            sensory_motor_summary = (
                f"Could not load sensory-motor trace due to error: {repr(exc)}"
            )

        refinement_prompt = prompt_4_refine_policy(
            current_policy_code=current_policy_code,
            current_mean_reward=current_mean_reward,
            current_std_reward=current_std_reward,
            best_policy_code=best_policy_code,
            best_mean_reward=best_mean_reward,
            sensory_motor_summary=sensory_motor_summary,
            previous_policy_code=previous_policy_code,
            previous_mean_reward=previous_mean_reward,
        )

        prompt_path = dirs["prompts_dir"] / f"{experiment_id}_iter{iteration:03d}_prompt4.txt"
        save_text(prompt_path, refinement_prompt)

        try:
            response = query_ollama(
                prompt=refinement_prompt,
                model=model,
                temperature=REFINEMENT_TEMPERATURE,
                num_predict=MAX_RESPONSE_TOKENS,
            )

            response_path = dirs["prompts_dir"] / (
                f"{experiment_id}_iter{iteration:03d}_response4_raw.txt"
            )
            save_text(response_path, response)

            improved_policy_code = extract_python_code(response)

            policy_path = dirs["policies_dir"] / f"{experiment_id}_iter{iteration:03d}_policy.py"
            save_text(policy_path, improved_policy_code)

            trace_path = dirs["traces_dir"] / f"{experiment_id}_iter{iteration:03d}_trace.jsonl"

            result = evaluate_generated_policy(
                code=improved_policy_code,
                env_id=ENV_ID,
                n_episodes=N_EVAL_EPISODES,
                seed=SEED + 1000 + iteration,
                trace_path=trace_path,
            )

            result["experiment_id"] = experiment_id
            result["iteration"] = iteration
            result["model"] = model
            result["model_safe"] = model_safe
            result["temperature"] = REFINEMENT_TEMPERATURE
            result["policy_path"] = str(policy_path)
            result["trace_path"] = str(trace_path)
            result["prompt_path"] = str(prompt_path)

            is_valid = result["n_failures"] == 0
            is_best = is_valid and result["mean_reward"] > best_mean_reward

            if is_best:
                best_policy_code = improved_policy_code
                best_mean_reward = float(result["mean_reward"])
                best_std_reward = float(result["std_reward"])
                best_iteration = iteration

            eval_path = dirs["eval_logs_dir"] / f"{experiment_id}_iter{iteration:03d}_eval.json"
            save_json(eval_path, result)

            try:
                last_20 = load_last_n_trace_steps(trace_path, n=LAST_N_TRACE_STEPS)
                save_json(
                    dirs["traces_dir"] / f"{experiment_id}_iter{iteration:03d}_last20.json",
                    {"last_20_steps": last_20},
                )
            except Exception:
                pass

            row = {
                "experiment_id": experiment_id,
                "iteration": iteration,
                "source": "refinement",
                "model": model,
                "model_safe": model_safe,
                "temperature": REFINEMENT_TEMPERATURE,
                "env_id": ENV_ID,
                "mean_reward": result["mean_reward"],
                "std_reward": result["std_reward"],
                "min_reward": result["min_reward"],
                "max_reward": result["max_reward"],
                "total_env_steps": result["total_env_steps"],
                "n_failures": result["n_failures"],
                "best_mean_reward_so_far": best_mean_reward,
                "best_iteration_so_far": best_iteration,
                "is_best": is_best,
                "status": "valid" if is_valid else "runtime_failures",
                "policy_path": str(policy_path),
                "trace_path": str(trace_path),
                "prompt_path": str(prompt_path),
                "eval_path": str(eval_path),
            }

            append_refinement_summary(row, model)

            print(
                f"Iteration {iteration} reward: "
                f"{result['mean_reward']:.3f} +/- {result['std_reward']:.3f}"
            )
            print(f"Failures: {result['n_failures']}")
            print(f"Best reward so far: {best_mean_reward:.3f} at iteration {best_iteration}")

            previous_policy_code = current_policy_code
            previous_mean_reward = current_mean_reward

            if is_best:
                current_policy_code = improved_policy_code
                current_mean_reward = float(result["mean_reward"])
                current_std_reward = float(result["std_reward"])
                current_trace_path = trace_path
            else:
                current_policy_code = best_policy_code
                current_mean_reward = best_mean_reward
                current_std_reward = best_std_reward

        except Exception as exc:
            print(f"Iteration {iteration} failed for model {model}: {repr(exc)}")

            row = {
                "experiment_id": experiment_id,
                "iteration": iteration,
                "source": "refinement",
                "model": model,
                "model_safe": model_safe,
                "temperature": REFINEMENT_TEMPERATURE,
                "env_id": ENV_ID,
                "mean_reward": None,
                "std_reward": None,
                "min_reward": None,
                "max_reward": None,
                "total_env_steps": 0,
                "n_failures": None,
                "best_mean_reward_so_far": best_mean_reward,
                "best_iteration_so_far": best_iteration,
                "is_best": False,
                "status": "invalid_generation",
                "error": repr(exc),
                "policy_path": "",
                "trace_path": "",
                "prompt_path": str(prompt_path),
                "eval_path": "",
            }

            append_refinement_summary(row, model)
            continue

    print("\n" + "=" * 80)
    print(f"Refinement completed for model: {model}")
    print(f"Best reward: {best_mean_reward:.3f} +/- {best_std_reward:.3f}")
    print(f"Best iteration: {best_iteration}")
    print(f"Model summary path: {dirs['summary_path']}")
    print("=" * 80)


def main() -> None:
    for model in MODELS:
        run_refinement_for_model(model)

    print("\n" + "#" * 80)
    print("ALL MODEL REFINEMENT RUNS COMPLETED")
    print(f"Global summary: {GLOBAL_REFINEMENT_SUMMARY}")
    print("#" * 80)

    if GLOBAL_REFINEMENT_SUMMARY.exists():
        df = pd.read_csv(GLOBAL_REFINEMENT_SUMMARY)
        valid_df = df[df["status"] == "valid"].copy()

        if not valid_df.empty:
            summary = (
                valid_df.groupby("model")
                .agg(
                    best_reward=("mean_reward", "max"),
                    mean_reward=("mean_reward", "mean"),
                    n_valid=("mean_reward", "count"),
                    best_iteration=("best_iteration_so_far", "max"),
                )
                .reset_index()
            )

            print("\nRefinement summary by model:")
            print(summary.to_string(index=False))


if __name__ == "__main__":
    main()