import math

def policy(obs):
    cos_theta = float(obs[0])
    sin_theta = float(obs[1])
    theta_dot = float(obs[2])

    theta = math.atan2(sin_theta, cos_theta)

    # Constants for proportional gains
    k1 = 0.5
    k2 = 0.1
    k3 = 1.0
    k4 = 0.5
    k5 = 0.01

    # Thresholds for angular velocity
    threshold1 = 5.0
    threshold2 = 2.0

    torque = 0.0

    if abs(cos_theta) > 0.5 or abs(sin_theta) > 0.5:
        # Far from upright
        torque += k1 * theta
    elif abs(theta_dot) > threshold1:
        # Rotating too quickly
        torque -= k3 * theta_dot
    elif abs(theta_dot) > threshold2:
        # Moderate angular velocity
        torque -= k4 * theta_dot
    else:
        # Near upright and low angular velocity
        torque += k5 * sin_theta

    # Ensure the torque is within the valid range [-2.0, 2.0]
    torque = max(-2.0, min(2.0, torque))

    return [float(torque)]