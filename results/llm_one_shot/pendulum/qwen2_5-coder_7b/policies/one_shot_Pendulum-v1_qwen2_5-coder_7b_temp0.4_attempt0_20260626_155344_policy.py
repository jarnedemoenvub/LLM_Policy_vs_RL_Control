import math

def policy(obs):
    cos_theta = float(obs[0])
    sin_theta = float(obs[1])
    theta_dot = float(obs[2])

    theta = math.atan2(sin_theta, cos_theta)

    # Correcting orientation
    if cos(theta) > 0 and sin(theta) > 0:
        torque_orientation = -0.5 * (cos(theta) + sin(theta))
    elif cos(theta) < 0 and sin(theta) < 0:
        torque_orientation = 0.5 * (cos(theta) + sin(theta))
    else:
        torque_orientation = 0

    # Minimizing angular velocity
    if theta_dot > 0:
        torque_angular_velocity = -theta_dot / abs(theta_dot)
    elif theta_dot < 0:
        torque_angular_velocity = theta_dot / abs(theta_dot)
    else:
        torque_angular_velocity = 0

    # Balancing control effort
    if cos(theta) > 0.9 and sin(theta) > 0.9:
        torque_balance = 0.1 * (cos(theta) + sin(theta))
    elif cos(theta) < -0.9 and sin(theta) < -0.9:
        torque_balance = -0.1 * (cos(theta) + sin(theta))
    else:
        torque_balance = 0.5 * (cos(theta) + sin(theta))

    # Combining corrections
    torque = torque_orientation + torque_angular_velocity + torque_balance

    if abs(theta_dot) > 2:
        torque /= 2

    torque = max(-2.0, min(2.0, torque))

    return [float(torque)]