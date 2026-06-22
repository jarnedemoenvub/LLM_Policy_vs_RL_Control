import math

def policy(obs):
    cos_theta = float(obs[0])
    sin_theta = float(obs[1])
    theta_dot = float(obs[2])

    theta = math.atan2(sin_theta, cos_theta)

    if abs(theta) > 3.14/4: # far from upright
        torque_magnitude = -0.5 * (cos_theta + sin_theta)
        torque = max(-2.0, min(2.0, -torque_magnitude))
        
    elif theta_dot > 3 or theta_dot < -3: # rapid rotation
        torque = max(-2.0, min(2.0, -theta_dot/2))
    
    else: 
        torque = -2.0 * (cos_theta + sin_theta) / 100
    
    return [float(torque)]