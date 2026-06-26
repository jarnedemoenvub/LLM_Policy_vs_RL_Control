import math

def policy(obs):
    cos_theta = float(obs[0])
    sin_theta = float(obs[1])
    theta_dot = float(obs[2])

    # Calculate angle in radians
    theta = math.atan2(sin_theta, cos_theta)

    # Main torque formula: balance between correcting the pendulum's position and damping its velocity
    # We use a combination of sine and cosine to correct the pendulum's position, and a simple damping term for the velocity
    # The signs are chosen such that when the pendulum is upright (theta close to 0), we apply less torque to avoid oscillations
    torque = -2 * math.sin(theta) + 1.5 * theta_dot

    # Clip the torque within the valid range [-2, 2]
    torque = max(-2.0, min(2.0, torque))

    return [float(torque)]