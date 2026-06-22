import math

def policy(obs):
    cos_theta = float(obs[0])
    sin_theta = float(obs[1])
    theta_dot = float(obs[2])

    theta = math.atan2(sin_theta, cos_theta)

    if abs(cos_theta) > 0.7 or abs(sin_theta) > 0.7:
        # Pendulum far from upright
        torque = -cos_theta * 2 if cos_theta < -0.7 else (cos_theta + 1) * 2 if cos_theta > 0.7 else -sin_theta * 2 if sin_theta < -0.7 else (sin_theta + 1) * 2
    elif abs(theta_dot) > 6:
        # Pendulum rotating too quickly
        torque = -theta_dot * 1.5 if theta_dot < -6 else (theta_dot + 6) * 0.75
    else:
        # Pendulum near upright
        torque = -theta_dot * 0.5 if theta_dot < -4 else (theta_dot + 4) * 0.25

    torque = max(-2.0, min(2.0, torque))

    return [float(torque)]