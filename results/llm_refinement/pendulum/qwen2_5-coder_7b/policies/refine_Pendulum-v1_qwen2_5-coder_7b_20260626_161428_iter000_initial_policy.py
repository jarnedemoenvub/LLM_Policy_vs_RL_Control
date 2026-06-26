import math

def policy(obs):
    cos_theta = float(obs[0])
    sin_theta = float(obs[1])
    theta_dot = float(obs[2])

    theta = math.atan2(sin_theta, cos_theta)

    torque = -theta * 2.0 - theta_dot * 0.5

    torque = max(-2.0, min(2.0, torque))

    return [float(torque)]