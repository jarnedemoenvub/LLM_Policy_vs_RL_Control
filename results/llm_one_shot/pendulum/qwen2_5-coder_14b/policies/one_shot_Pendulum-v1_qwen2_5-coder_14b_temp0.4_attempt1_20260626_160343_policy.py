import math

def policy(obs):
    cos_theta = float(obs[0])
    sin_theta = float(obs[1])
    theta_dot = float(obs[2])

    theta = math.atan2(sin_theta, cos_theta)

    # Calculate the torque based on orientation and angular velocity
    if abs(cos_theta) > 0.5 or abs(sin_theta) > 0.5:
        torque = -2.0 * theta - 0.5 * theta_dot
    else:
        torque = -1.0 * theta - 0.3 * theta_dot

    # Clip the torque to the range [-2.0, 2.0]
    torque = max(-2.0, min(2.0, torque))

    return [float(torque)]