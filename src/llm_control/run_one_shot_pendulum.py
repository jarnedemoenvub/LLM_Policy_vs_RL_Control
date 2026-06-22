from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from src.llm_control.code_extractor import extract_python_code
from src.llm_control.config import (
    ENV_ID,
    EVAL_LOGS_DIR,
    MAX_RESPONSE_TOKENS,
    N_EVAL_EPISODES,
    OLLAMA_MODEL,
    POLICIES_DIR,
    PROMPTS_DIR,
    SEED,
    TEMPERATURE,
    TRACES_DIR,
)
from src.llm_control.evaluate_policy import evaluate_generated_policy, load_last_n_trace_steps
from src.llm_control.ollama_client import query_ollama
from src.llm_control.prompts import (
    prompt_1_strategy,
    prompt_2_rules,
    prompt_3_code,
)


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"one_shot_{ENV_ID}_{OLLAMA_MODEL.replace(':', '_')}_{timestamp}"

    print(f"Run ID: {run_id}")

    # -------------------------
    # Prompt 1: high-level strategy
    # -------------------------
    p1 = prompt_1_strategy()
    save_text(PROMPTS_DIR / f"{run_id}_prompt1_strategy.txt", p1)

    strategy_response = query_ollama(
        prompt=p1,
        model=OLLAMA_MODEL,
        temperature=TEMPERATURE,
        num_predict=MAX_RESPONSE_TOKENS,
    )
    save_text(PROMPTS_DIR / f"{run_id}_response1_strategy.txt", strategy_response)

    print("Prompt 1 completed.")

    # -------------------------
    # Prompt 2: IF-THEN-ELSE rules
    # -------------------------
    p2 = prompt_2_rules(strategy_response)
    save_text(PROMPTS_DIR / f"{run_id}_prompt2_rules.txt", p2)

    rules_response = query_ollama(
        prompt=p2,
        model=OLLAMA_MODEL,
        temperature=TEMPERATURE,
        num_predict=MAX_RESPONSE_TOKENS,
    )
    save_text(PROMPTS_DIR / f"{run_id}_response2_rules.txt", rules_response)

    print("Prompt 2 completed.")

    # -------------------------
    # Prompt 3: Python code
    # -------------------------
    p3 = prompt_3_code(strategy_response, rules_response)
    save_text(PROMPTS_DIR / f"{run_id}_prompt3_code.txt", p3)

    code_response = query_ollama(
        prompt=p3,
        model=OLLAMA_MODEL,
        temperature=TEMPERATURE,
        num_predict=MAX_RESPONSE_TOKENS,
    )
    save_text(PROMPTS_DIR / f"{run_id}_response3_code_raw.txt", code_response)

    code = extract_python_code(code_response)
    policy_path = POLICIES_DIR / f"{run_id}_policy.py"
    save_text(policy_path, code)

    print(f"Policy saved to: {policy_path}")

    # -------------------------
    # Evaluate generated policy
    # -------------------------
    trace_path = TRACES_DIR / f"{run_id}_trace.jsonl"

    result = evaluate_generated_policy(
        code=code,
        env_id=ENV_ID,
        n_episodes=N_EVAL_EPISODES,
        seed=SEED,
        trace_path=trace_path,
    )

    result["run_id"] = run_id
    result["model"] = OLLAMA_MODEL
    result["temperature"] = TEMPERATURE
    result["policy_path"] = str(policy_path)
    result["trace_path"] = str(trace_path)

    result_path = EVAL_LOGS_DIR / f"{run_id}_eval.json"
    save_json(result_path, result)

    print("Evaluation completed.")
    print(json.dumps(result, indent=2))

    # -------------------------
    # Store last 20 sensory-motor steps separately
    # -------------------------
    last_20 = load_last_n_trace_steps(trace_path, n=20)
    save_json(
        TRACES_DIR / f"{run_id}_last20_sensory_motor_steps.json",
        {"last_20_steps": last_20},
    )

    # -------------------------
    # Append to summary CSV
    # -------------------------
    summary_path = EVAL_LOGS_DIR.parent / "summary.csv"

    row = {
        "run_id": run_id,
        "env_id": ENV_ID,
        "model": OLLAMA_MODEL,
        "temperature": TEMPERATURE,
        "n_eval_episodes": N_EVAL_EPISODES,
        "seed": SEED,
        "mean_reward": result["mean_reward"],
        "std_reward": result["std_reward"],
        "min_reward": result["min_reward"],
        "max_reward": result["max_reward"],
        "total_env_steps": result["total_env_steps"],
        "n_failures": result["n_failures"],
        "policy_path": str(policy_path),
        "trace_path": str(trace_path),
    }

    if summary_path.exists():
        df = pd.read_csv(summary_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(summary_path, index=False)

    print(f"Summary updated: {summary_path}")


if __name__ == "__main__":
    main()