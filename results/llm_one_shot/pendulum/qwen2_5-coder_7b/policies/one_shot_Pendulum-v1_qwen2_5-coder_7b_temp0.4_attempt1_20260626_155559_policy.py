import math

def policy(obs):
    cos_theta = float(obs[0])
    sin_theta = float(obs[1])
    theta_dot = float(obs[2])

    theta = math.atan2(sin_theta, cos_theta)

    K_p = 1.5
    K_i = 0.01
    K_d = 0.05

    if not hasattr(policy, 'integral_error'):
        policy.integral_error = 0

    e = -theta
    policy.integral_error += e
    derivative_error = -theta_dot

    torque = K_p * e + K_i * policy.integral_error + K_d * derivative_error

    torque = max(-2.0, min(2.0, torque))

    return [float(torque)]