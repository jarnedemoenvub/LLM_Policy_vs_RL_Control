import math

def policy(obs):
    cos_theta = float(obs[0])
    sin_theta = float(obs[1])
    theta_dot = float(obs[2])

    # Determine if the pendulum is far from upright (rule 1)
    if abs(cos_theta) > 0.8 or abs(sin_theta) > 0.8:
        # Apply a large torque in the opposite direction of the pendulum
        torque = -1.5 * math.copysign(1, theta_dot)
    
    else:  # Pendulum is near upright (rule 2 and rule 3)
        if abs(theta_dot) > 4:  # Rotating too quickly
            torque = -1.0  # Increase damping torque
        else:
            torque = -0.5 * theta_dot  # Apply small, opposing torque
    
    torque = max(-2.0, min(2.0, torque))  # Clip torque to valid range

    return [float(torque)]