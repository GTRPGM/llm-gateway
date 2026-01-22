import asyncio

from llm_gateway.core.config import settings
from llm_gateway.schemas.chat import ChatMessage, ChatRequest
from llm_gateway.services.providers.gemini import GeminiProvider


async def test_real_gemini_call():
    print("--- Real Gemini API Call Test (Sequential) ---")

    # 0. Check for API Key
    assert settings.GOOGLE_API_KEY, "GOOGLE_API_KEY not found in settings."

    provider = GeminiProvider()

    # 1. Basic Chat Test
    print("\n[Step 1] Testing Basic Chat...")
    request = ChatRequest(
        model="gemini-2.0-flash-light",
        messages=[
            ChatMessage(
                role="user",
                content="TRPG의 GM(Game Master)이 하는 일 3가지만 짧게 알려줘.",
            )
        ],
        temperature=0.7,
    )

    response = await provider.chat_complete(request)
    assert response.choices[0].message.content, "Empty response from Gemini."
    print("Result: SUCCESS")
    print(f"Response: {response.choices[0].message.content[:100]}...")

    # 2. JSON Mode Test
    print("\n[Step 2] Testing JSON Mode (Structured Output)...")
    request_json = ChatRequest(
        model="gemini-2.0-flash-light",
        messages=[
            ChatMessage(
                role="system",
                content=(
                    "당신은 TRPG 주사위 판정기입니다. "
                    "결과는 반드시 JSON으로만 출력하세요."
                ),
            ),
            ChatMessage(
                role="user",
                content=(
                    "근력 15인 캐릭터가 난이도 12의 문을 부수려고 합니다. "
                    "성공 여부를 판정해줘."
                ),
            ),
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )

    response_json = await provider.chat_complete(request_json)
    assert response_json.choices[0].message.content, "Empty JSON response from Gemini."
    # JSON 형식인지 간단히 검증
    assert "{" in response_json.choices[0].message.content, (
        "Response is not in JSON format."
    )

    print("Result: SUCCESS")
    print(f"JSON Response: {response_json.choices[0].message.content}")

    print("\n--- All Tests Passed Successfully ---")


if __name__ == "__main__":
    asyncio.run(test_real_gemini_call())
