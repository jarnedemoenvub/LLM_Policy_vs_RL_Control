import math

def policy(obs):
    cos_theta = float(obs[0])
    sin_theta = float(obs[1])
    theta_dot = float(obs[2])

    # Calculate the angle theta from the observation components
    theta = math.atan2(sin_theta, cos_theta)

    # Proportional and derivative gains
    Kp = 5.0
    Kd = 0.5

    # Calculate torque based on proportional and derivative terms
    torque = -Kp * sin(theta) - Kd * theta_dot

    # Clip the torque to the valid range [-2.0, 2.0]
    torque = max(-2.0, min(2.0, torque))

    return [torque]