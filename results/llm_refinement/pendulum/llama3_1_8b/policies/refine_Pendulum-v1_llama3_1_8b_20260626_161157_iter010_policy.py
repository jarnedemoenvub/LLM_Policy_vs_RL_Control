import math

def policy(obs):
    cos_theta = float(obs[0])
    sin_theta = float(obs[1])
    theta_dot = float(obs[2])

    # Calculate theta from cos and sin values
    theta = math.atan2(sin_theta, cos_theta)

    # Main torque formula using a combination of angle correction and velocity damping
    torque = -5.0 * (theta + 0.25 * math.sin(theta)) - 1.5 * theta_dot

    # Optional clipping to ensure the torque is within the valid range
    torque = max(-2.0, min(2.0, torque))

    return [float(torque)]