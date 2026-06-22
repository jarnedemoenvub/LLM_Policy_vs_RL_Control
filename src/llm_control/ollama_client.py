import requests


def query_ollama(
    prompt: str,
    model: str,
    temperature: float = 0.0,
    num_predict: int = 2048,
) -> str:
    """
    Sends a prompt to a local Ollama model and returns the text response.
    Ollama must be running locally.
    """
    url = "http://localhost:11434/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }

    response = requests.post(url, json=payload, timeout=300)
    response.raise_for_status()

    data = response.json()
    return data["response"]