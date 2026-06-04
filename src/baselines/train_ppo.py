import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from src.envs.make_env import make_env


def train_ppo(env_id="Pendulum-v1", seed=42, total_timesteps=300_000):
    os.makedirs("results/raw", exist_ok=True)

    env = make_env(env_id)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        seed=seed,
        verbose=1,
    )

    model.learn(total_timesteps=total_timesteps)

    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=10,
        deterministic=True,
    )

    model_path = f"results/raw/ppo_{env_id}_seed{seed}"
    model.save(model_path)

    result = {
        "method": "PPO",
        "env_id": env_id,
        "seed": seed,
        "total_timesteps": total_timesteps,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
    }

    df = pd.DataFrame([result])
    csv_path = f"results/raw/ppo_{env_id}_seed{seed}.csv"
    df.to_csv(csv_path, index=False)

    env.close()

    print("\nPPO evaluation")
    print("--------------")
    print(f"Environment: {env_id}")
    print(f"Seed: {seed}")
    print(f"Timesteps: {total_timesteps}")
    print(f"Mean reward: {mean_reward:.2f}")
    print(f"Std reward: {std_reward:.2f}")
    print(f"Saved model to: {model_path}.zip")
    print(f"Saved results to: {csv_path}")


if __name__ == "__main__":
    train_ppo()