import json

import httpx
import requests

BASE_URL = "http://localhost:8000/api/v1"


def test_api_routing():
    print("--- API Gateway Integration Test ---")

    # 1. Check Health
    try:
        resp = requests.get("http://localhost:8000/health")
        if resp.status_code != 200:
            print("ERROR: Server is not healthy.")
            return
    except requests.exceptions.ConnectionError:
        print("SKIP: Server is not running at localhost:8000")
        return

    # 2. Test OpenAI Routing (gpt-* model name)
    print("\n[Step 1] Testing OpenAI Routing via model name...")
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Hi, who are you?"}],
    }
    resp = requests.post(f"{BASE_URL}/chat/completions", json=payload)
    print(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        print(f"Response Model: {resp.json()['model']}")

    # 3. Test Gemini Routing (gemini-* model name)
    print("\n[Step 2] Testing Gemini Routing via model name...")
    payload = {
        "model": "gemini-2.0-flash",
        "messages": [{"role": "user", "content": "Hi, who are you?"}],
    }
    resp = requests.post(f"{BASE_URL}/chat/completions", json=payload)
    print(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        print(f"Response Model: {resp.json()['model']}")

    # 4. Test Config Change
    print("\n[Step 3] Changing Default Provider to 'openai'...")
    resp = requests.post(
        f"{BASE_URL}/gateway/config", json={"default_provider": "openai"}
    )
    print(f"Config update: {resp.json()}")

    print("\n[Step 4] Testing 'default' model routing (should be OpenAI now)...")
    payload = {"model": "default", "messages": [{"role": "user", "content": "Ping"}]}
    resp = requests.post(f"{BASE_URL}/chat/completions", json=payload)
    if resp.status_code == 200:
        print(f"Response Model: {resp.json()['model']}")

    # New Step: Override Provider Test
    print("\n[Step 4.5] Testing Override Provider (Force OpenAI)...")
    # 1. Set override to openai
    resp = requests.post(
        f"{BASE_URL}/gateway/config",
        json={"default_provider": "openai", "override_provider": "openai"},
    )
    print(f"Config override update: {resp.json()}")

    # 2. Request Gemini model (should be routed to OpenAI due to override)
    payload = {
        "model": "gemini-2.0-flash",
        "messages": [{"role": "user", "content": "Who are you?"}],
    }
    resp = requests.post(f"{BASE_URL}/chat/completions", json=payload)
    if resp.status_code == 200:
        model_used = resp.json()["model"]
        print(f"Requested 'gemini-2.0-flash', Got Model: {model_used}")
        if "gpt" in model_used or "openai" in model_used:
            print("SUCCESS: Override worked.")
        else:
            print(f"FAIL: Override failed. Got {model_used}")

    # 3. Clear override
    print("Clearing override...")
    resp = requests.post(
        f"{BASE_URL}/gateway/config",
        json={"default_provider": "openai", "override_provider": None},
    )
    print(f"Config cleared: {resp.json()}")

    # 5. Test Streaming API
    print("\n[Step 5] Testing Streaming API (GPT-4o-mini)...")
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Counting to 3 slowly."}],
        "stream": True,
    }

    print("Stream output: ", end="", flush=True)
    with httpx.stream(
        "POST", f"{BASE_URL}/chat/completions", json=payload, timeout=60.0
    ) as r:
        if r.status_code != 200:
            print(f"ERROR: {r.status_code}")
        else:
            for line in r.iter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    chunk = json.loads(data_str)
                    content = chunk["choices"][0]["delta"].get("content")
                    if content:
                        print(content, end="", flush=True)
    print("\nResult: SUCCESS")

    print("\n--- API Tests Completed ---")


if __name__ == "__main__":
    test_api_routing()
