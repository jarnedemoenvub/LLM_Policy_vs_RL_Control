import numpy as np


def controller(obs):
    """
    Simple hand-written controller for Pendulum-v1.

    Observation:
        obs[0] = cos(theta)
        obs[1] = sin(theta)
        obs[2] = angular velocity

    Action:
        torque in [-2, 2]
    """
    cos_theta = obs[0]
    sin_theta = obs[1]
    angular_velocity = obs[2]

    theta = np.arctan2(sin_theta, cos_theta)

    torque = -2.0 * theta - 0.5 * angular_velocity

    torque = np.clip(torque, -2.0, 2.0)

    return np.array([torque], dtype=np.float32)


def random_controller(obs):
    """
    Random controller for Pendulum-v1.

    Pendulum-v1 expects one continuous action:
    torque in the range [-2.0, 2.0].
    """
    return np.array([np.random.uniform(-2.0, 2.0)], dtype=np.float32)