import math

def policy(obs):
    """
    Improved controller for Pendulum-v1 environment.

    Args:
        obs (list): Observation vector of length 3: [cos(theta), sin(theta), theta_dot]

    Returns:
        list: Action vector of length 1: [torque]
    """

    # Extract observation components
    cos_theta = float(obs[0])
    sin_theta = float(obs[1])
    theta_dot = float(obs[2])

    # Calculate angle (theta) from direction cosines
    theta = math.atan2(sin_theta, cos_theta)

    # Define controller gains
    Kp = 4.0  # Proportional gain for angle correction
    Kd = 3.5  # Derivative gain for velocity damping

    # Calculate torque using a smooth continuous formula
    torque = -Kp * theta - Kd * theta_dot

    # Clip torque to valid range [-2, 2]
    torque = max(-2.0, min(2.0, torque))

    return [float(torque)]