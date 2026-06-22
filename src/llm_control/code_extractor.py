import re


def extract_python_code(response: str) -> str:
    """
    Extracts the first Python code block from an LLM response.
    If no fenced code block exists, returns the full response.
    """
    pattern = r"```(?:python)?\s*(.*?)```"
    match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)

    if match:
        return match.group(1).strip()

    return response.strip()