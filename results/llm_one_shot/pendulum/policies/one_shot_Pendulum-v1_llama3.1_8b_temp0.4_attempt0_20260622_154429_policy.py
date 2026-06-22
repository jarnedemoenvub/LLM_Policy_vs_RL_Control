import math

def policy(obs):
    cos_theta = float(obs[0])
    sin_theta = float(obs[1])
    theta_dot = float(obs[2])

    if abs(cos_theta) > 0.7:
        # Pendulum far from upright: Apply a torque to correct the pendulum's direction
        theta = math.atan2(sin_theta, cos_theta)
        torque = -2.0 * theta - 0.5 * theta_dot
        return [float(max(-2.0, min(2.0, torque)))]

    elif abs(theta_dot) > 4:
        # Pendulum rotating too quickly: Increase the magnitude of the stabilizing torque to slow down the pendulum
        torque = -0.1 * theta_dot + 0.5 * cos_theta
        return [float(max(-2.0, min(2.0, torque)))]

    else:
        # Pendulum near upright and rotating slowly: Apply a small, stabilizing torque that counteracts any remaining angular momentum
        torque = -0.01 * theta_dot + 0.05 * cos_theta
        return [float(max(-2.0, min(2.0, torque)))]