import math

def policy(obs):
    cos_theta = float(obs[0])
    sin_theta = float(obs[1])
    theta_dot = float(obs[2])

    theta = math.atan2(sin_theta, cos_theta)

    if abs(cos_theta) < 0.7 and abs(sin_theta) < 0.7:
        torque = -0.5 * theta_dot
    else:
        torque = -2.0 * theta - 0.5 * theta_dot

    torque = max(-2.0, min(2.0, torque))

    return [float(torque)]