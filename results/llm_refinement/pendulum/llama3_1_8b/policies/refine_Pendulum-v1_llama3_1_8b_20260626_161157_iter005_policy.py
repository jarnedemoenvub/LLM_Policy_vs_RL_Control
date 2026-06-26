import math

def policy(obs):
    """
    A smooth continuous controller for Pendulum-v1 environment.

    Args:
        obs (list): Observation vector of length 3: [cos(theta), sin(theta), theta_dot].

    Returns:
        list: Action vector of length 1: [torque].
    """

    # Convert observation to float
    cos_theta = float(obs[0])
    sin_theta = float(obs[1])
    theta_dot = float(obs[2])

    # Calculate angle in radians
    theta = math.atan2(sin_theta, cos_theta)

    # Define coefficients for angle correction and velocity damping
    Kp = 4.0  # Proportional gain (strength of angle correction)
    Kd = -1.5  # Derivative gain (strength of velocity damping)

    # Calculate torque using theta and theta_dot
    torque = Kp * theta + Kd * theta_dot

    # Clip torque to valid range [-2, 2]
    torque = max(-2.0, min(2.0, torque))

    return [torque]