import math

def policy(obs):
    cos_theta = float(obs[0])
    sin_theta = float(obs[1])
    theta_dot = float(obs[2])

    if abs(cos_theta) < 0.5 and abs(theta_dot) > 3:
        torque = -0.5 * theta_dot
    elif abs(cos_theta) >= 0.5 or (abs(theta_dot) <= 3 and cos_theta != 0):
        torque = -2.0 * math.atan2(sin_theta, cos_theta) - 0.5 * theta_dot

    torque = max(-2.0, min(2.0, torque))

    return [float(torque)]