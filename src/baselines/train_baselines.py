import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from stable_baselines3 import DDPG, PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
import config

def train_ppo(env_id="Pendulum-v1", seed=42, total_timesteps=100_000):
    os.makedirs("results/baselines/ppo", exist_ok=True)

    env = gym.make(env_id)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        seed=seed,
        device=config.DEVICE
    )

    model.learn(total_timesteps=total_timesteps)

    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=config.N_EVAL_EPISODES,
        deterministic=True,
    )

    model_path = f"results/baselines/ppo/ppo_{env_id}"
    model.save(model_path)

    result = {
        "method": "PPO",
        "env_id": env_id,
        "total_timesteps": total_timesteps,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
    }

    df = pd.DataFrame([result])
    csv_path = f"results/baselines/ppo/ppo_{env_id}_seed{seed}.csv"
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


def train_sac(env_id="Pendulum-v1", seed=42, total_timesteps=100_000):
    os.makedirs("results/baselines/sac", exist_ok=True)

    env = gym.make(env_id)

    model = SAC(
        policy="MlpPolicy",
        env=env,
        seed=seed,
        device=config.DEVICE
    )

    model.learn(total_timesteps=total_timesteps)

    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=config.N_EVAL_EPISODES,
        deterministic=True,
    )

    model_path = f"results/baselines/sac/sac_{env_id}"
    model.save(model_path)

    result = {
        "method": "SAC",
        "env_id": env_id,
        "total_timesteps": total_timesteps,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
    }

    df = pd.DataFrame([result])
    csv_path = f"results/baselines/sac/sac_{env_id}_seed{seed}.csv"
    df.to_csv(csv_path, index=False)

    env.close()

    print("\nSAC evaluation")
    print("--------------")
    print(f"Environment: {env_id}")
    print(f"Seed: {seed}")
    print(f"Timesteps: {total_timesteps}")
    print(f"Mean reward: {mean_reward:.2f}")
    print(f"Std reward: {std_reward:.2f}")
    print(f"Saved model to: {model_path}.zip")
    print(f"Saved results to: {csv_path}")


def train_ddpg(env_id="Pendulum-v1", seed=42, total_timesteps=100_000):
    os.makedirs("results/baselines/ddpg", exist_ok=True)

    env = gym.make(env_id)

    model = DDPG(
        policy="MlpPolicy",
        env=env,
        seed=seed,
        device=config.DEVICE
    )

    model.learn(total_timesteps=total_timesteps)

    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=config.N_EVAL_EPISODES,
        deterministic=True
    )

    model_path = f"results/baselines/ddpg/ddpg_{env_id}"
    model.save(model_path)

    result = {
        "method": "DDPG",
        "env_id": env_id,
        "total_timesteps": total_timesteps,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
    }

    df = pd.DataFrame([result])
    csv_path = f"results/baselines/ddpg/ddpg_{env_id}_seed{seed}.csv"
    df.to_csv(csv_path, index=False)

    env.close()

    print("\nDDPG evaluation")
    print("--------------")
    print(f"Environment: {env_id}")
    print(f"Seed: {seed}")
    print(f"Timesteps: {total_timesteps}")
    print(f"Mean reward: {mean_reward:.2f}")
    print(f"Std reward: {std_reward:.2f}")
    print(f"Saved model to: {model_path}.zip")
    print(f"Saved results to: {csv_path}")


if __name__ == "__main__":
    for i in range(config.N_RUNS):
        seed = config.SEED + i
        train_ppo(env_id="Pendulum-v1", seed=seed, total_timesteps=config.PPO_TIMESTEPS)
        train_sac(env_id="Pendulum-v1", seed=seed, total_timesteps=config.SAC_TIMESTEPS)
        train_ddpg(env_id="Pendulum-v1", seed=seed, total_timesteps=config.DDPG_TIMESTEPS)