PENDULUM_TASK_DESCRIPTION = """
You are designing a controller for the Gymnasium environment Pendulum-v1.

The agent controls a pendulum by applying torque to its joint.

Observation vector:
The observation is a continuous vector of length 3:

obs[0] = cos(theta), range [-1, 1]
obs[1] = sin(theta), range [-1, 1]
obs[2] = theta_dot, angular velocity, range approximately [-8, 8]

The values cos(theta) and sin(theta) encode the direction of the pendulum.
Examples:
[1, 0] means the pendulum points to the right.
[0, 1] means the pendulum points up.
[-1, 0] means the pendulum points to the left.
[0, -1] means the pendulum points down.

Action vector:
The action is a continuous vector of length 1:

action[0] = torque applied to the pendulum, range [-2, 2].

The controller must return a valid action as a list or NumPy array containing one float.
The goal is to keep the pendulum upright while minimizing angular velocity and control effort.
Higher reward is better. Pendulum rewards are usually negative, and good policies obtain rewards closer to zero.

Important:
The final controller must be a deterministic Python function with this exact signature:

def policy(obs):
    ...
    return [torque]

The function receives one observation vector and returns one action vector.
"""


def prompt_1_strategy() -> str:
    return f"""
{PENDULUM_TASK_DESCRIPTION}

Prompt 1: High-level control strategy

Explain a high-level control strategy for this task.

Do not write Python code yet.
Focus on how the controller should use:
- cos(theta)
- sin(theta)
- angular velocity theta_dot
- torque direction and magnitude

Explain how the policy should behave when:
- the pendulum is far from upright
- the pendulum is near upright
- the pendulum is rotating too quickly
"""


def prompt_2_rules(strategy_text: str) -> str:
    return f"""
{PENDULUM_TASK_DESCRIPTION}

The following high-level strategy was proposed:

{strategy_text}

Prompt 2: IF-THEN-ELSE rules

Translate the strategy into clear IF-THEN-ELSE control rules.

Do not write Python code yet.
The rules should be unambiguous enough to be converted into a Python function.

The rules must produce one continuous torque value in the range [-2, 2].
"""


def prompt_3_code(strategy_text: str, rules_text: str) -> str:
    return f"""
{PENDULUM_TASK_DESCRIPTION}

High-level strategy:
{strategy_text}

IF-THEN-ELSE rules:
{rules_text}

Prompt 3: Python controller

Convert the rules into executable Python code.

Requirements:
- Return only one Python code block.
- Define exactly one function named policy.
- The function signature must be: def policy(obs):
- The function must return a list containing one float: [torque]
- The torque must always be clipped to the range [-2.0, 2.0].
- You may use only the math module and basic Python operations.
- Do not import os, sys, subprocess, pathlib, socket, requests, shutil, or any unsafe library.
- Do not read or write files.
- Do not use input(), eval(), exec(), compile(), open(), globals(), locals(), or __import__().
- Do not include explanations outside the code block.
- The policy function must work independently when called with any valid observation.

Very important: You must define theta before using it: theta = math.atan2(sin_theta, cos_theta)
- Never use a variable before assigning it.

Example required format:

```python
import math

def policy(obs):
    cos_theta = float(obs[0])
    sin_theta = float(obs[1])
    theta_dot = float(obs[2])

    theta = math.atan2(sin_theta, cos_theta)

    torque = -2.0 * theta - 0.5 * theta_dot

    torque = max(-2.0, min(2.0, torque))

    return [float(torque)]"""