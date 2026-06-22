import math

def policy(obs):
    cos_theta = float(obs[0])
    sin_theta = float(obs[1])
    theta_dot = float(obs[2])

    if abs(cos_theta) < 0.5 or abs(sin_theta) < 0.5:
        if theta_dot > 2: 
            torque = -math.copysign(0.5 * abs(theta_dot), theta_dot)
        elif (cos_theta <= 0 and sin_theta >= 0) or (cos_theta >= 0 and sin_theta <= 0):
            torque = math.copysign(1, cos_theta)
        else:
            torque = max(0.5 * abs(theta_dot), 1)
    else:
        if (cos_theta <= 0 and sin_theta >= 0) or (cos_theta >= 0 and sin_theta <= 0):
            torque = -math.copysign(0.2, theta_dot)
        else:
            torque = max(0.1 * abs(theta_dot), 0)

    torque = max(-2.0, min(2.0, torque))

    return [float(torque)]