import math

def policy(obs):
    cos_theta = float(obs[0])
    sin_theta = float(obs[1])
    theta_dot = float(obs[2])

    # Calculate the angle theta from the observation components
    theta = math.atan2(sin_theta, cos_theta)

    # Calculate angular error from upright position (theta = 0)
    angular_error = theta

    # Calculate torque based on angular error and angular velocity
    torque = -2.0 * angular_error - 0.5 * theta_dot

    # Clip the torque to the valid range [-2.0, 2.0]
    torque = max(-2.0, min(2.0, torque))

    return [torque]