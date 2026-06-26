import math

def policy(obs):
    cos_theta = float(obs[0])
    sin_theta = float(obs[1])
    theta_dot = float(obs[2])

    theta = math.atan2(sin_theta, cos_theta)

    Kp = 5.0
    Kd = 1.0
    Ks = 0.5

    torque = -Kp * theta - Kd * theta_dot + Ks * sin_theta * abs(theta_dot)

    torque = max(-2.0, min(2.0, torque))

    return [torque]