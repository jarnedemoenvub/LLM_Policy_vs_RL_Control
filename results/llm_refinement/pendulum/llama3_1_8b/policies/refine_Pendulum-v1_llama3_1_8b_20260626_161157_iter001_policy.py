import math

def policy(obs):
    # Extract values from the observation vector
    cos_theta = float(obs[0])
    sin_theta = float(obs[1])
    theta_dot = float(obs[2])

    # Calculate theta using arctangent of sin and cos
    theta = math.atan2(sin_theta, cos_theta)

    # Define coefficients for angle correction (Kp) and velocity damping (Kd)
    Kp = 0.5  # Strength of the angle correction
    Kd = 0.25  # Strength of the velocity damping

    # Calculate torque using a smooth continuous formula
    torque = -Kp * theta - Kd * theta_dot

    # Clip torque to ensure it's within the valid range [-2, 2]
    torque = max(-2.0, min(2.0, torque))

    return [float(torque)]