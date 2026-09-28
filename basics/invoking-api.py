import os
import json
import urllib.request
from pathlib import Path

def load_env(env_path: Path = None):
    """
    Load environment variables from .env file.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path or Path(__file__).resolve().parent.parent / ".env")
        return
    except ImportError:
        pass

    if env_path is None:
        env_path = Path(__file__).resolve().parent.parent / ".env"

    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip().strip("'\""))


def invoke_openrouter_http(prompt: str, model: str = "google/gemma-4-26b-a4b-it:free") -> str:
    """
    Invoke a model using Python's standard library (urllib.request).
    No pip dependencies required!
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set. Please check your .env file.")

    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:3000",  # Optional for rankings on OpenRouter
        "X-Title": "AI Practice",                # Optional
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST"
    )

    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        raise RuntimeError(f"OpenRouter API error ({e.code}): {error_body}") from e


def invoke_openrouter_openai_sdk(prompt: str, model: str = "google/gemma-4-26b-a4b-it:free") -> str:
    """
    Alternative: Invoke using the official openai Python library.
    Requires: pip install openai
    """
    from openai import OpenAI

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set. Please check your .env file.")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content, response.usage.total_tokens


if __name__ == "__main__":
    # 1. Load API Key from .env
    load_env()
    print("-"*50)
    test_prompt = input("Enter your prompt: ")
    print("-"*50)
    # Popular free models:
    # - google/gemma-4-31b-it:free
    # - google/gemma-4-26b-a4b-it:free
    # - nvidia/nemotron-3-super-120b-a12b:free
    # - nvidia/nemotron-3.5-lightning:free
    model_name = "nvidia/nemotron-3-super-120b-a12b:free"

    print(f"Sending prompt to OpenRouter model: {model_name}\n")
    print(f"Prompt: {test_prompt}\n")

    # Using the openai library call
    try:
        reply, tokens = invoke_openrouter_openai_sdk(prompt=test_prompt, model=model_name)
        print("Response from OpenRouter:")
        print("-" * 50)
        print(reply)
        print(f"Total tokens used: {tokens}")
        print("-" * 50)
    except Exception as e:
        print(f"Error: {e}")
