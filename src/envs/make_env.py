import gymnasium as gym


def make_env(env_id: str, render_mode=None):
    """
    Create a Gymnasium environment.

    For Week 1 we start with Pendulum-v1 because it has:
    - continuous observations
    - continuous actions
    - simple dynamics
    """
    return gym.make(env_id, render_mode=render_mode)