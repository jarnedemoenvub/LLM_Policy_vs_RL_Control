from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.llm_control.evaluate_policy import evaluate_generated_policy


PROJECT_ROOT = Path.cwd()

ENV_ID = "Pendulum-v1"
N_EVAL_EPISODES = 100
SEED = 42

ONE_SHOT_SUMMARY = (
    PROJECT_ROOT
    / "results"
    / "llm_one_shot"
    / "pendulum"
    / "summary.csv"
)

REFINEMENT_SUMMARY = (
    PROJECT_ROOT
    / "results"
    / "llm_refinement"
    / "pendulum"
    / "refinement_summary_all_models.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "llm_final_eval_100"
    / "pendulum"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def make_safe_name(text: str) -> str:
    return (
        str(text)
        .replace(":", "_")
        .replace("/", "_")
        .replace(".", "_")
        .replace(" ", "_")
    )


def load_summaries() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not ONE_SHOT_SUMMARY.exists():
        raise FileNotFoundError(f"Could not find one-shot summary: {ONE_SHOT_SUMMARY}")

    if not REFINEMENT_SUMMARY.exists():
        raise FileNotFoundError(f"Could not find refinement summary: {REFINEMENT_SUMMARY}")

    one_shot_df = pd.read_csv(ONE_SHOT_SUMMARY)
    refine_df = pd.read_csv(REFINEMENT_SUMMARY)

    return one_shot_df, refine_df


def select_best_one_shot_per_model(one_shot_df: pd.DataFrame) -> pd.DataFrame:
    valid_one_shot = one_shot_df[one_shot_df["status"] == "valid"].copy()

    if valid_one_shot.empty:
        raise RuntimeError("No valid one-shot policies found.")

    best_one_shot = (
        valid_one_shot
        .sort_values("mean_reward", ascending=False)
        .groupby("model", as_index=False)
        .first()
    )

    best_one_shot["selection_type"] = "best_one_shot"
    best_one_shot["iteration"] = 0
    best_one_shot["source_id"] = best_one_shot["run_id"]

    return best_one_shot[
        [
            "selection_type",
            "model",
            "source_id",
            "iteration",
            "temperature",
            "mean_reward",
            "std_reward",
            "policy_path",
        ]
    ].rename(
        columns={
            "mean_reward": "previous_mean_reward_10ep",
            "std_reward": "previous_std_reward_10ep",
        }
    )


def select_best_refined_per_model(refine_df: pd.DataFrame) -> pd.DataFrame:
    valid_refine = refine_df[refine_df["status"] == "valid"].copy()
    valid_refine = valid_refine[valid_refine["source"] == "refinement"].copy()

    if valid_refine.empty:
        raise RuntimeError("No valid refined policies found.")

    best_refined = (
        valid_refine
        .sort_values("mean_reward", ascending=False)
        .groupby("model", as_index=False)
        .first()
    )

    best_refined["selection_type"] = "best_refined"
    best_refined["source_id"] = best_refined["experiment_id"]

    return best_refined[
        [
            "selection_type",
            "model",
            "source_id",
            "iteration",
            "temperature",
            "mean_reward",
            "std_reward",
            "policy_path",
        ]
    ].rename(
        columns={
            "mean_reward": "previous_mean_reward_10ep",
            "std_reward": "previous_std_reward_10ep",
        }
    )


def evaluate_selected_policies(policies_to_eval: pd.DataFrame) -> pd.DataFrame:
    final_rows = []

    for _, row in policies_to_eval.iterrows():
        model = row["model"]
        selection_type = row["selection_type"]
        iteration = int(row["iteration"])
        policy_path = Path(row["policy_path"])

        if not policy_path.exists():
            print(f"Skipping missing policy: {policy_path}")
            continue

        code = policy_path.read_text(encoding="utf-8")

        model_safe = make_safe_name(model)
        selection_safe = make_safe_name(selection_type)

        eval_id = f"{selection_safe}_{model_safe}_iter{iteration}_100ep"

        trace_path = OUTPUT_DIR / f"{eval_id}_trace.jsonl"
        result_path = OUTPUT_DIR / f"{eval_id}_eval.json"

        print("\n" + "=" * 80)
        print(f"Evaluating: {eval_id}")
        print(f"Model: {model}")
        print(f"Type: {selection_type}")
        print(f"Previous 10-episode mean: {row['previous_mean_reward_10ep']:.3f}")
        print(f"Policy path: {policy_path}")
        print("=" * 80)

        result = evaluate_generated_policy(
            code=code,
            env_id=ENV_ID,
            n_episodes=N_EVAL_EPISODES,
            seed=SEED,
            trace_path=trace_path,
        )

        result["eval_id"] = eval_id
        result["selection_type"] = selection_type
        result["model"] = model
        result["iteration"] = iteration
        result["temperature"] = row["temperature"]
        result["previous_mean_reward_10ep"] = row["previous_mean_reward_10ep"]
        result["previous_std_reward_10ep"] = row["previous_std_reward_10ep"]
        result["policy_path"] = str(policy_path)
        result["trace_path"] = str(trace_path)
        result["status"] = "valid" if result["n_failures"] == 0 else "runtime_failures"

        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

        final_rows.append(
            {
                "eval_id": eval_id,
                "selection_type": selection_type,
                "model": model,
                "iteration": iteration,
                "temperature": row["temperature"],
                "previous_mean_reward_10ep": row["previous_mean_reward_10ep"],
                "previous_std_reward_10ep": row["previous_std_reward_10ep"],
                "mean_reward_100ep": result["mean_reward"],
                "std_reward_100ep": result["std_reward"],
                "min_reward_100ep": result["min_reward"],
                "max_reward_100ep": result["max_reward"],
                "n_eval_episodes": N_EVAL_EPISODES,
                "total_env_steps": result["total_env_steps"],
                "n_failures": result["n_failures"],
                "status": result["status"],
                "policy_path": str(policy_path),
                "trace_path": str(trace_path),
                "eval_json_path": str(result_path),
            }
        )

        print(
            f"100-episode result: {result['mean_reward']:.3f} "
            f"+/- {result['std_reward']:.3f}"
        )
        print(f"Failures: {result['n_failures']}")

    return pd.DataFrame(final_rows)


def make_comparison(final_eval_df: pd.DataFrame) -> pd.DataFrame:
    comparison = final_eval_df.pivot_table(
        index="model",
        columns="selection_type",
        values="mean_reward_100ep",
        aggfunc="first",
    ).reset_index()

    if "best_one_shot" in comparison.columns and "best_refined" in comparison.columns:
        comparison["refinement_improvement_100ep"] = (
            comparison["best_refined"] - comparison["best_one_shot"]
        )

    return comparison


def main() -> None:
    print("Project root:", PROJECT_ROOT)
    print("One-shot summary:", ONE_SHOT_SUMMARY)
    print("Refinement summary:", REFINEMENT_SUMMARY)
    print("Output directory:", OUTPUT_DIR)

    one_shot_df, refine_df = load_summaries()

    best_one_shot = select_best_one_shot_per_model(one_shot_df)
    best_refined = select_best_refined_per_model(refine_df)

    policies_to_eval = pd.concat(
        [best_one_shot, best_refined],
        ignore_index=True,
    )

    selected_path = OUTPUT_DIR / "selected_policies_for_100ep_eval.csv"
    policies_to_eval.to_csv(selected_path, index=False)

    print("\nSelected policies for 100-episode evaluation:")
    print(
        policies_to_eval[
            [
                "selection_type",
                "model",
                "iteration",
                "temperature",
                "previous_mean_reward_10ep",
                "policy_path",
            ]
        ].to_string(index=False)
    )

    final_eval_df = evaluate_selected_policies(policies_to_eval)

    final_eval_path = OUTPUT_DIR / "final_100_episode_evaluation.csv"
    final_eval_df.to_csv(final_eval_path, index=False)

    comparison = make_comparison(final_eval_df)
    comparison_path = OUTPUT_DIR / "one_shot_vs_refined_100ep_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    pretty = final_eval_df[
        [
            "model",
            "selection_type",
            "iteration",
            "temperature",
            "previous_mean_reward_10ep",
            "mean_reward_100ep",
            "std_reward_100ep",
            "min_reward_100ep",
            "max_reward_100ep",
            "n_failures",
            "status",
        ]
    ].copy()

    pretty = pretty.sort_values(["model", "selection_type"])
    pretty = pretty.round(3)

    print("\nFinal 100-episode evaluation:")
    print(pretty.to_string(index=False))

    print("\nOne-shot vs refined comparison:")
    print(comparison.round(3).to_string(index=False))

    print("\nSaved files:")
    print(f"- {selected_path}")
    print(f"- {final_eval_path}")
    print(f"- {comparison_path}")


if __name__ == "__main__":
    main()