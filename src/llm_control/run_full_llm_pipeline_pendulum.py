from __future__ import annotations

import subprocess
import sys
import time


def run_command(command: list[str]) -> None:
    print("\n" + "=" * 80)
    print("Running command:")
    print(" ".join(command))
    print("=" * 80)

    result = subprocess.run(command)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with return code {result.returncode}: {' '.join(command)}"
        )


def main() -> None:
    start_time = time.time()

    print("\nStarting full LLM control pipeline for Pendulum-v1")

    # Step 1: run one-shot policy generation for all configured models
    run_command([
        sys.executable,
        "-m",
        "src.llm_control.run_multiple_one_shot_pendulum",
    ])

    # Step 2: run iterative refinement for all configured models
    run_command([
        sys.executable,
        "-m",
        "src.llm_control.run_iterative_refinement_pendulum",
    ])

    elapsed = time.time() - start_time

    print("\n" + "=" * 80)
    print("Full LLM pipeline completed successfully.")
    print(f"Total time: {elapsed / 60:.2f} minutes")
    print("=" * 80)


if __name__ == "__main__":
    main()