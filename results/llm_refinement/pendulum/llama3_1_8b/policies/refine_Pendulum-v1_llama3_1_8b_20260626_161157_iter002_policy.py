import math

def policy(obs):
    cos_theta = float(obs[0])
    sin_theta = float(obs[1])
    theta_dot = float(obs[2])

    # Calculate theta from sin and cos values
    theta = math.atan2(sin_theta, cos_theta)

    # Define coefficients for angle correction and velocity damping
    Kp = 1.5
    Kd = 0.7

    # Main torque formula: combine angle correction and velocity damping
    torque = -Kp * theta - Kd * theta_dot

    # Optional clipping to ensure torque is within valid range [-2, 2]
    torque = max(-2.0, min(2.0, torque))

    return [float(torque)]