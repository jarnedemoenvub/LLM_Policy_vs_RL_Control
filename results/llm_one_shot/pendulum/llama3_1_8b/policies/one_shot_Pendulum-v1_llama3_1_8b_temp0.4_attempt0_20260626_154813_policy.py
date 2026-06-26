import math

def policy(obs):
    cos_theta = float(obs[0])
    sin_theta = float(obs[1])
    theta_dot = float(obs[2])

    # Ensure that the angle is within the range of [-pi, pi]
    theta = math.atan2(sin_theta, cos_theta)
    if theta > math.pi:
        theta -= 2 * math.pi

    # Rule 3: Rotating too quickly
    if abs(theta_dot) > 4:
        torque = -0.5 * theta_dot  # Prioritize reducing angular velocity over correcting the pendulum's direction

    # Rule 1: Far from upright
    elif abs(cos_theta) < 0.7:
        torque = -2.0 * theta  # Apply torque in a direction that corrects the pendulum's orientation to bring it closer to the upright position

    # Rule 2: Near upright
    else:
        torque = -0.5 * theta_dot + 0.1 * cos_theta  # Prioritize maintaining stability while allowing for some angular velocity

    # Clip the torque value to the range [-2, 2]
    torque = max(-2.0, min(2.0, torque))

    return [float(torque)]