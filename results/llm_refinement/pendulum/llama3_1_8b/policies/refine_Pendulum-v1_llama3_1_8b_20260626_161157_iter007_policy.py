import math

def policy(obs):
    """
    A smooth continuous controller for the Gymnasium environment Pendulum-v1.

    Parameters:
    obs (list): Observation vector of length 3: [cos(theta), sin(theta), theta_dot]

    Returns:
    list: Action vector of length 1: [torque]
    """

    # Unpack observation vector
    cos_theta = float(obs[0])
    sin_theta = float(obs[1])
    theta_dot = float(obs[2])

    # Calculate theta using atan2 function to handle the case where theta is pi or -pi
    theta = math.atan2(sin_theta, cos_theta)

    # Define gains for angle correction and velocity damping
    Kp = 0.5  # Proportional gain (angle correction)
    Kd = 1.0  # Derivative gain (velocity damping)

    # Calculate torque using the main formula
    torque = -Kp * theta - Kd * theta_dot

    # Clip torque to ensure it's within the valid range [-2, 2]
    torque = max(-2.0, min(2.0, torque))

    return [float(torque)]