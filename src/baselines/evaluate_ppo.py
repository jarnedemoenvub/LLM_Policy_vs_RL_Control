import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from src.envs.make_env import make_env


def evaluate_ppo(
    model_path="results/raw/ppo_Pendulum-v1_seed42.zip",
    env_id="Pendulum-v1",
    episodes=100,
    seed=42,
):
    os.makedirs("results/baselines", exist_ok=True)

    env = make_env(env_id)
    env.reset(seed=seed)

    model = PPO.load(model_path)

    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=episodes,
        deterministic=True,
    )

    result = {
        "method": "PPO",
        "env_id": env_id,
        "episodes": episodes,
        "seed": seed,
        "model_path": model_path,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
    }

    df = pd.DataFrame([result])

    output_path = "results/baselines/ppo_Pendulum-v1_seed42_eval100.csv"
    df.to_csv(output_path, index=False)

    env.close()

    print("\nPPO Evaluation")
    print("--------------")
    print(df)
    print(f"\nSaved PPO result to: {output_path}")


if __name__ == "__main__":
    evaluate_ppo()