import math

def policy(obs):
    cos_theta = float(obs[0])
    sin_theta = float(obs[1])
    theta_dot = float(obs[2])

    sin_theta = 2 * sin_theta - 1
    cos_theta = 2 * cos_theta - 1

    if (sin_theta < -0.7 or sin_theta > 0.7) and abs(theta_dot) > 4:
        torque = 1 if theta_dot > 0 else -1
    elif (abs(sin_theta) < 0.3) and (theta_dot > 0):
        torque = max(-2, min(2, -0.5 * theta_dot))
    elif abs(theta_dot) > 4:
        torque = 1 if theta_dot < 0 else -1
        sin_theta = 2 * sin_theta - 1
        cos_theta = 2 * cos_theta - 1
        theta = math.atan2(sin_theta, cos_theta)
        torque -= 0.5 * theta / (abs(theta) + 1e-8)
    elif (abs(sin_theta) < 0.3) and (theta_dot < 2):
        torque = max(-2, min(2, -0.5 * theta_dot))
    
    torque = max(-2.0, min(2.0, torque))

    return [float(torque)]