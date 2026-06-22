import math

def policy(obs):
    cos_theta = float(obs[0])
    sin_theta = float(obs[1])
    theta_dot = float(obs[2])

    # Rule 1: Pendulum far from upright
    if abs(cos_theta) > 0.9 or abs(sin_theta) > 0.9:
        if cos_theta < -0.9:  # pendulum points up
            torque = max(-2.0, min(2.0, -2))
        elif sin_theta > 0.9:  # pendulum points down
            torque = max(-2.0, min(2.0, 2))
        else:
            torque = max(-2.0, min(2.0, -1 * cos_theta))

    # Rule 2: Pendulum near upright
    elif abs(cos_theta) < 0.9 and abs(sin_theta) < 0.9:
        if cos_theta > 0.9:  # pendulum points up
            torque = max(-2.0, min(2.0, 0.5))
        elif sin_theta < -0.9:  # pendulum points down
            torque = max(-2.0, min(2.0, -0.5))
        else:
            torque = max(-2.0, min(2.0, 0.5 * (1 - theta_dot**2)))

    # Rule 3: Pendulum rotating too quickly
    elif abs(theta_dot) > 4:
        if cos_theta > 0.9:  # pendulum points up
            torque = max(-2.0, min(2.0, -1 * theta_dot))
        elif sin_theta < -0.9:  # pendulum points down
            torque = max(-2.0, min(2.0, 1 * theta_dot))
        else:
            torque = max(-2.0, min(2.0, -1 * theta_dot * (1 - abs(theta_dot)/4)))

    # Default case: apply a small stabilizing torque
    else:
        torque = 0

    return [float(torque)]