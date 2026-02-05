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

    print("\n--- API Tests Completed ---")


if __name__ == "__main__":
    test_api_routing()
