# src/llm_control/run_multiple_one_shot_pendulum.py

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from src.llm_control.code_extractor import extract_python_code
from src.llm_control.config import (
    ENV_ID,
    MAX_RESPONSE_TOKENS,
    N_EVAL_EPISODES,
    SEED,
    TEMPERATURE,
    MODELS
)
from src.llm_control.evaluate_policy import evaluate_generated_policy, load_last_n_trace_steps
from src.llm_control.ollama_client import query_ollama
from src.llm_control.prompts import (
    prompt_1_strategy,
    prompt_2_rules,
    prompt_3_code,
)


# Choose the models, temperatures, and repetitions here

RUNS_PER_TEMPERATURE = 3


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_RESULTS_DIR = PROJECT_ROOT / "results" / "llm_one_shot" / "pendulum"
GLOBAL_SUMMARY_PATH = BASE_RESULTS_DIR / "summary.csv"


def make_safe_model_name(model: str) -> str:
    return model.replace(":", "_").replace("/", "_").replace(".", "_")


def get_model_dirs(model: str) -> dict[str, Path]:
    model_safe = make_safe_model_name(model)
    model_dir = BASE_RESULTS_DIR / model_safe

    return {
        "model_dir": model_dir,
        "prompts_dir": model_dir / "prompts",
        "policies_dir": model_dir / "policies",
        "eval_logs_dir": model_dir / "eval_logs",
        "traces_dir": model_dir / "sensory_motor_traces",
        "model_summary_path": model_dir / "summary.csv",
    }


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def append_csv(row: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        df = pd.read_csv(path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(path, index=False)


def append_summary(row: dict, model: str) -> None:
    dirs = get_model_dirs(model)

    # Global summary with all models
    append_csv(row, GLOBAL_SUMMARY_PATH)

    # Separate summary for this model
    append_csv(row, dirs["model_summary_path"])


def run_single_attempt(model: str, temperature: float, attempt_idx: int) -> dict:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_name_safe = make_safe_model_name(model)
    dirs = get_model_dirs(model)

    run_id = (
        f"one_shot_{ENV_ID}_{model_name_safe}"
        f"_temp{temperature}_attempt{attempt_idx}_{timestamp}"
    )

    print("\n" + "=" * 80)
    print(f"Starting run: {run_id}")
    print(f"Model: {model}")
    print(f"Temperature: {temperature}")
    print("=" * 80)

    try:
        # -------------------------
        # Prompt 1: high-level strategy
        # -------------------------
        p1 = prompt_1_strategy()
        save_text(dirs["prompts_dir"] / f"{run_id}_prompt1_strategy.txt", p1)

        strategy_response = query_ollama(
            prompt=p1,
            model=model,
            temperature=temperature,
            num_predict=MAX_RESPONSE_TOKENS,
        )
        save_text(dirs["prompts_dir"] / f"{run_id}_response1_strategy.txt", strategy_response)

        print("Prompt 1 completed.")

        # -------------------------
        # Prompt 2: IF-THEN-ELSE rules
        # -------------------------
        p2 = prompt_2_rules(strategy_response)
        save_text(dirs["prompts_dir"] / f"{run_id}_prompt2_rules.txt", p2)

        rules_response = query_ollama(
            prompt=p2,
            model=model,
            temperature=temperature,
            num_predict=MAX_RESPONSE_TOKENS,
        )
        save_text(dirs["prompts_dir"] / f"{run_id}_response2_rules.txt", rules_response)

        print("Prompt 2 completed.")

        # -------------------------
        # Prompt 3: Python code
        # -------------------------
        p3 = prompt_3_code(strategy_response, rules_response)
        save_text(dirs["prompts_dir"] / f"{run_id}_prompt3_code.txt", p3)

        code_response = query_ollama(
            prompt=p3,
            model=model,
            temperature=temperature,
            num_predict=MAX_RESPONSE_TOKENS,
        )
        save_text(dirs["prompts_dir"] / f"{run_id}_response3_code_raw.txt", code_response)

        code = extract_python_code(code_response)
        policy_path = dirs["policies_dir"] / f"{run_id}_policy.py"
        save_text(policy_path, code)

        print(f"Policy saved to: {policy_path}")

        # -------------------------
        # Evaluate generated policy
        # -------------------------
        trace_path = dirs["traces_dir"] / f"{run_id}_trace.jsonl"

        result = evaluate_generated_policy(
            code=code,
            env_id=ENV_ID,
            n_episodes=N_EVAL_EPISODES,
            seed=SEED + attempt_idx,
            trace_path=trace_path,
        )

        result["run_id"] = run_id
        result["model"] = model
        result["model_safe"] = model_name_safe
        result["temperature"] = temperature
        result["attempt_idx"] = attempt_idx
        result["policy_path"] = str(policy_path)
        result["trace_path"] = str(trace_path)
        result["status"] = "valid" if result["n_failures"] == 0 else "runtime_failures"

        result_path = dirs["eval_logs_dir"] / f"{run_id}_eval.json"
        save_json(result_path, result)

        # Store final 20 sensory-motor steps for later refinement
        last_20 = load_last_n_trace_steps(trace_path, n=20)
        save_json(
            dirs["traces_dir"] / f"{run_id}_last20_sensory_motor_steps.json",
            {"last_20_steps": last_20},
        )

        row = {
            "run_id": run_id,
            "env_id": ENV_ID,
            "model": model,
            "model_safe": model_name_safe,
            "temperature": temperature,
            "attempt_idx": attempt_idx,
            "n_eval_episodes": N_EVAL_EPISODES,
            "seed": SEED + attempt_idx,
            "mean_reward": result["mean_reward"],
            "std_reward": result["std_reward"],
            "min_reward": result["min_reward"],
            "max_reward": result["max_reward"],
            "total_env_steps": result["total_env_steps"],
            "n_failures": result["n_failures"],
            "status": result["status"],
            "policy_path": str(policy_path),
            "trace_path": str(trace_path),
            "eval_path": str(result_path),
        }

        append_summary(row, model)

        print("Evaluation completed.")
        print(
            f"Mean reward: {result['mean_reward']:.3f} "
            f"+/- {result['std_reward']:.3f}"
        )
        print(f"Failures: {result['n_failures']}")
        print(f"Status: {result['status']}")

        return row

    except Exception as exc:
        print(f"Run failed before/during evaluation: {repr(exc)}")

        row = {
            "run_id": run_id,
            "env_id": ENV_ID,
            "model": model,
            "model_safe": model_name_safe,
            "temperature": temperature,
            "attempt_idx": attempt_idx,
            "n_eval_episodes": N_EVAL_EPISODES,
            "seed": SEED + attempt_idx,
            "mean_reward": None,
            "std_reward": None,
            "min_reward": None,
            "max_reward": None,
            "total_env_steps": 0,
            "n_failures": None,
            "status": "invalid_generation",
            "error": repr(exc),
            "policy_path": "",
            "trace_path": "",
            "eval_path": "",
        }

        append_summary(row, model)
        return row


def main() -> None:
    all_rows = []

    for model in MODELS:
        print("\n" + "#" * 80)
        print(f"RUNNING MODEL: {model}")
        print("#" * 80)

        for attempt_idx in range(RUNS_PER_TEMPERATURE):
            row = run_single_attempt(
                model=model,
                temperature=TEMPERATURE,
                attempt_idx=attempt_idx,
            )
            all_rows.append(row)

            # Small pause so timestamps differ and Ollama gets a tiny break
            time.sleep(1)

    print("\n" + "=" * 80)
    print("All runs completed.")
    print("=" * 80)

    df = pd.DataFrame(all_rows)
    valid_df = df[df["status"] == "valid"].copy()

    if not valid_df.empty:
        print("\nValid run summary by model and temperature:")
        summary = (
            valid_df.groupby(["model", "temperature"])
            .agg(
                n_valid=("run_id", "count"),
                mean_reward_mean=("mean_reward", "mean"),
                mean_reward_std=("mean_reward", "std"),
                best_reward=("mean_reward", "max"),
                worst_reward=("mean_reward", "min"),
            )
            .reset_index()
        )

        print(summary.to_string(index=False))
    else:
        print("No valid runs found.")

    print(f"\nGlobal summary CSV: {GLOBAL_SUMMARY_PATH}")

    print("\nModel-specific folders:")
    for model in MODELS:
        print(f"- {model}: {get_model_dirs(model)['model_dir']}")


if __name__ == "__main__":
    main()