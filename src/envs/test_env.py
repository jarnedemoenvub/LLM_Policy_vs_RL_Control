from src.envs.make_env import make_env


def main():
    env = make_env("Pendulum-v1")

    obs, info = env.reset(seed=42)

    print("Initial observation:")
    print(obs)

    print("\nObservation space:")
    print(env.observation_space)

    print("\nAction space:")
    print(env.action_space)

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    print("\nAfter one random action:")
    print("Observation:", obs)
    print("Reward:", reward)
    print("Terminated:", terminated)
    print("Truncated:", truncated)

    env.close()


if __name__ == "__main__":
    main()