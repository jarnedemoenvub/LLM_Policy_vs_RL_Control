import math

def policy(obs):
    """
    Returns a valid action as a list or NumPy array containing one float.
    
    Parameters:
    obs (list): Observation vector of length 3: [cos(theta), sin(theta), theta_dot]
    
    Returns:
    list: Action vector of length 1: [torque applied to the pendulum, range [-2.0, 2.0]]
    """
    cos_theta = float(obs[0])
    sin_theta = float(obs[1])
    theta_dot = float(obs[2])

    # Calculate theta using math.atan2
    theta = math.atan2(sin_theta, cos_theta)

    # Main torque formula: torque = -Kp * theta - Kd * theta_dot
    # Signs and coefficients chosen based on reward and sensory-motor data
    Kp = 1.5  # Angle correction strength
    Kd = 0.7  # Velocity damping strength

    # Calculate torque using main formula
    torque = -Kp * theta - Kd * theta_dot

    # Clip torque to valid range [-2.0, 2.0]
    torque = max(-2.0, min(2.0, torque))

    return [float(torque)]