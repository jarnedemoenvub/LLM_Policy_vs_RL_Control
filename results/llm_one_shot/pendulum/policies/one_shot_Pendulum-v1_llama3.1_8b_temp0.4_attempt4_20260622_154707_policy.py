import math

def policy(obs):
    cos_theta = float(obs[0])
    sin_theta = float(obs[1])
    theta_dot = float(obs[2])

    if abs(cos_theta) > 0.8 or abs(sin_theta) > 0.8:
        # Far from upright, prioritize rapid correction
        torque = -2.0 * math.atan2(-sin_theta, -cos_theta) - 0.5 * theta_dot
    elif abs(theta_dot) > 4:
        # Rotating too quickly, prioritize slowing down rotation
        torque = -2.0 * math.sign(theta_dot)
    else:
        # Near upright or stable, fine-tune alignment and reduce velocity
        torque = -1.0 * math.atan2(-sin_theta, cos_theta) - 0.5 * theta_dot

    torque = max(-2.0, min(2.0, torque))

    return [float(torque)]