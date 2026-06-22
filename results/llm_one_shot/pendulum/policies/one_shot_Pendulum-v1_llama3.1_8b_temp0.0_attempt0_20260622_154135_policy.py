import math

def policy(obs):
    cos_theta = float(obs[0])
    sin_theta = float(obs[1])
    theta_dot = float(obs[2])

    if abs(cos_theta) > 0.7 or abs(sin_theta) > 0.7:
        if cos_theta < -0.7:
            torque = -2 * (1 + cos_theta)
        elif cos_theta > 0.7:
            torque = 2 * (1 - cos_theta)
        elif sin_theta < -0.7:
            torque = 2 * (1 - sin_theta)
        else:
            torque = -2 * (1 + sin_theta)

    elif abs(cos_theta) <= 0.3 and abs(sin_theta) <= 0.3:
        if theta_dot > 0:
            torque = -0.5 * theta_dot
        else:
            torque = 0.5 * abs(theta_dot)

    elif abs(theta_dot) > 4:
        if theta_dot > 0:
            torque = -1.5 * theta_dot
        else:
            torque = 1.5 * abs(theta_dot)

    else:
        torque = -2.0 * math.atan2(sin_theta, cos_theta) - 0.5 * theta_dot

    torque = max(-2.0, min(2.0, torque))

    return [float(torque)]