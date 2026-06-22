import math

def policy(obs):
    cos_theta = float(obs[0])
    sin_theta = float(obs[1])
    theta_dot = float(obs[2])

    if abs(cos_theta) > 0.5 or abs(sin_theta) > 0.5:
        error = cos_theta * sin_theta
        torque = max(-2.0, min(2.0, -2.0 * error))
    
    elif abs(theta_dot) > 4.0:
        torque = max(-2.0, min(2.0, -1.5 * theta_dot))
    
    else:
        if cos_theta < 0 and sin_theta > 0:
            torque = -0.5
        elif cos_theta > 0 and sin_theta < 0:
            torque = 0.5
    
    return [float(torque)]