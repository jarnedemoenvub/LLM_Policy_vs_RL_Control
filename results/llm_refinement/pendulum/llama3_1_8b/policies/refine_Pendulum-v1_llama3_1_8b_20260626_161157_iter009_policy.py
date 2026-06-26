import math

def policy(obs):
    """
    Returns an action (torque) to control the pendulum in Pendulum-v1 environment.
    
    Parameters:
    obs (list): Observation vector of length 3. Contains cos(theta), sin(theta), and theta_dot.
    
    Returns:
    list: Action vector containing one float, which is the torque applied to the pendulum.
    """
    
    # Extract observation components
    cos_theta = float(obs[0])
    sin_theta = float(obs[1])
    theta_dot = float(obs[2])
    
    # Calculate angle (theta) in radians
    theta = math.atan2(sin_theta, cos_theta)
    
    # Define gains for angle and velocity correction
    Kp = 3.0  # Proportional gain for angle correction
    Kd = -1.5  # Derivative gain for velocity damping
    
    # Calculate torque based on angle and velocity correction
    torque = -Kp * theta - Kd * theta_dot
    
    # Clip torque to valid range [-2, 2]
    torque = max(-2.0, min(2.0, torque))
    
    return [float(torque)]