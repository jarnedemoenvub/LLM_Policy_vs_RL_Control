import math

def policy(obs):
    cos_theta = float(obs[0])
    sin_theta = float(obs[1])
    theta_dot = float(obs[2])

    if abs(theta_dot) > 4:
        torque = -math.sign(theta_dot) * 1.5
    elif (cos_theta < 0 or sin_theta != 0) and abs(theta_dot) > 1:
        torque = math.sign(sin_theta) * 2
    else:
        theta = math.atan2(sin_theta, cos_theta)
        torque = -2.0 * theta - 0.5 * theta_dot

    torque = max(-2.0, min(2.0, torque))

    return [float(torque)]