import math

def policy(obs):
    """
    Returns the torque to apply to the pendulum based on its current state.

    Args:
        obs (list): A list containing three elements: cos(theta), sin(theta), and theta_dot.

    Returns:
        list: A list containing one float, representing the torque to apply.
    """

    # Extract values from observation
    cos_theta = float(obs[0])
    sin_theta = float(obs[1])
    theta_dot = float(obs[2])

    # Calculate theta using atan2 to handle edge cases where sin(theta) or cos(theta) is zero
    theta = math.atan2(sin_theta, cos_theta)

    # Define the gains for angle correction and velocity damping
    Kp = 0.5  # Proportional gain (strength of angle correction)
    Kd = 1.0  # Derivative gain (strength of velocity damping)

    # Calculate torque using a simple PD controller formula
    torque = -Kp * theta - Kd * theta_dot

    # Clip the torque to ensure it's within the valid range [-2, 2]
    torque = max(-2.0, min(2.0, torque))

    return [float(torque)]