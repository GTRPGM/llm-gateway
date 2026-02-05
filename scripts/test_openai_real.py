import asyncio
import json

from llm_gateway.core.config import settings
from llm_gateway.extensions.providers.openai import OpenAIProvider
from llm_gateway.schemas.chat import ChatMessage, ChatRequest


async def test_real_openai_call():
    print("--- Real OpenAI API Call Test ---")

    # 0. Check for API Key
    if not settings.OPENAI_API_KEY:
        print("SKIP: OPENAI_API_KEY not found in settings.")
        return

    provider = OpenAIProvider()

    # 1. Basic Chat Test
    print("\n[Step 1] Testing Basic Chat (GPT-4o-mini)...")
    content = "TRPG에서 다이스 갓(Dice God)이라는 표현의 유래를 한 줄로 알려줘."
    request = ChatRequest(
        model="gpt-4o-mini",
        messages=[ChatMessage(role="user", content=content)],
        temperature=0.7,
    )

    response = await provider.chat_complete(request)
    assert response.choices[0].message.content, "Empty response from OpenAI."
    print("Result: SUCCESS")
    print(f"Response: {response.choices[0].message.content}")

    # 2. JSON Mode Test
    print("\n[Step 2] Testing JSON Mode...")
    request_json = ChatRequest(
        model="gpt-4o-mini",
        messages=[
            ChatMessage(
                role="system",
                content="You are a character generator. Output MUST be in JSON.",
            ),
            ChatMessage(
                role="user",
                content="Generate a simple warrior character name and stat(STR).",
            ),
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )

    response_json = await provider.chat_complete(request_json)
    assert response_json.choices[0].message.content, "Empty JSON response from OpenAI."

    try:
        data = json.loads(response_json.choices[0].message.content)
        print(f"Result: SUCCESS (Parsed JSON: {data})")
    except json.JSONDecodeError:
        error_msg = (
            "Result: FAILURE (Invalid JSON: "
            f"{response_json.choices[0].message.content})"
        )
        print(error_msg)

    # 3. Streaming Test
    print("\n[Step 3] Testing Streaming...")
    request_stream = ChatRequest(
        model="gpt-4o-mini",
        messages=[ChatMessage(role="user", content="Write a two-sentence poem.")],
        stream=True,
    )

    print("Stream output: ", end="", flush=True)
    response_stream = await provider.chat_complete(request_stream)
    async for chunk in response_stream:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)
    print("\nResult: SUCCESS")

    print("\n--- OpenAI Tests Completed ---")


if __name__ == "__main__":
    asyncio.run(test_real_openai_call())
