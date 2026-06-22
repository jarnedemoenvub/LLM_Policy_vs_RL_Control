import ast
import math
import multiprocessing as mp
from typing import Any, Callable

import numpy as np


FORBIDDEN_NAMES = {
    "eval",
    "exec",
    "compile",
    "open",
    "input",
    "globals",
    "locals",
    "vars",
    "dir",
    "getattr",
    "setattr",
    "delattr",
}

ALLOWED_MODULES = {
    "math": math,
    "numpy": np,
    "np": np,
}

FORBIDDEN_MODULES = {
    "os",
    "sys",
    "subprocess",
    "pathlib",
    "socket",
    "requests",
    "shutil",
    "pickle",
    "builtins",
}


class UnsafeCodeError(Exception):
    pass


def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    """
    Allow only safe imports inside generated policy code.
    """
    root_name = name.split(".")[0]

    if root_name in ALLOWED_MODULES:
        return ALLOWED_MODULES[root_name]

    raise ImportError(f"Import of module '{name}' is not allowed.")


def check_ast_safety(code: str) -> None:
    tree = ast.parse(code)

    function_names = [
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    ]

    if "policy" not in function_names:
        raise UnsafeCodeError("Generated code must define a function named policy.")

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                root_name = alias.name.split(".")[0]

                if root_name in FORBIDDEN_MODULES:
                    raise UnsafeCodeError(f"Forbidden import: {root_name}")

                if root_name not in ALLOWED_MODULES:
                    raise UnsafeCodeError(f"Import not allowed: {root_name}")

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in FORBIDDEN_NAMES:
                    raise UnsafeCodeError(f"Forbidden function call: {node.func.id}")

            if isinstance(node.func, ast.Attribute):
                if node.func.attr in FORBIDDEN_NAMES:
                    raise UnsafeCodeError(f"Forbidden attribute call: {node.func.attr}")


def load_policy_from_code(code: str) -> Callable[[Any], Any]:
    """
    Loads the generated policy function after AST checks.
    """
    check_ast_safety(code)

    safe_globals = {
        "__builtins__": {
            "__import__": safe_import,
            "abs": abs,
            "min": min,
            "max": max,
            "float": float,
            "int": int,
            "range": range,
            "len": len,
            "bool": bool,
        },
        "math": math,
        "np": np,
        "numpy": np,
    }

    safe_locals = {}

    exec(code, safe_globals, safe_locals)

    if "policy" not in safe_locals:
        raise UnsafeCodeError("No policy function found after execution.")

    return safe_locals["policy"]


def _call_policy_worker(code: str, obs: list[float], queue: mp.Queue) -> None:
    try:
        policy = load_policy_from_code(code)
        action = policy(obs)
        queue.put(("ok", action))
    except Exception as exc:
        queue.put(("error", repr(exc)))


def call_policy_with_timeout(
    code: str,
    obs: np.ndarray,
    timeout_seconds: float = 1.0,
) -> np.ndarray:
    """
    Calls generated policy in a separate process to prevent infinite loops.
    Returns a clipped Pendulum action.
    """
    queue: mp.Queue = mp.Queue()

    process = mp.Process(
        target=_call_policy_worker,
        args=(code, obs.astype(float).tolist(), queue),
    )

    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join()
        raise TimeoutError("Policy execution timed out.")

    if queue.empty():
        raise RuntimeError("Policy process returned no result.")

    status, result = queue.get()

    if status != "ok":
        raise RuntimeError(f"Policy execution failed: {result}")

    action = np.array(result, dtype=np.float32).reshape(-1)

    if action.shape[0] != 1:
        raise ValueError(f"Pendulum action must have shape (1,), got {action.shape}")

    action = np.clip(action, -2.0, 2.0)

    return action