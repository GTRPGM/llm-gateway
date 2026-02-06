from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from openai import OpenAIError

from llm_gateway.main import app

client = TestClient(app(), raise_server_exceptions=False)


def test_validation_error_logging():
    """
    잘못된 요청 본문을 보냈을 때 422 에러와 함께 상세 내용이 반환되는지 확인합니다.
    """
    # model 필드가 누락된 잘못된 요청
    response = client.post(
        "/api/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    assert "body" in data
    assert "messages" in data["body"]


@pytest.mark.asyncio
async def test_provider_generic_error_handling():
    """
    Provider 내부에서 예상치 못한 에러 발생 시 500 에러가 반환되는지 확인합니다.
    """
    # chat_complete 자체를 mocking
    with patch(
        "llm_gateway.extensions.providers.openai.OpenAIProvider.chat_complete",
        side_effect=Exception("Unexpected Mock Error"),
    ):
        response = client.post(
            "/api/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 500
        # detail은 FastAPI/Starlette에 의해 Internal Server Error로 치환될 수 있음
        assert "Internal Server Error" in response.json()["detail"]


@pytest.mark.asyncio
async def test_provider_api_error_handling():
    """
    OpenAI API 호출 실패 시 502 에러가 반환되는지 확인합니다.
    """
    # OpenAI SDK 내부 호출을 mocking (chat_complete 안의 create 호출)
    with patch(
        "openai.resources.chat.completions.AsyncCompletions.create",
        side_effect=OpenAIError("Quota Exceeded"),
    ):
        response = client.post(
            "/api/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 502
        assert "Error from OpenAI Provider" in response.json()["detail"]
        assert "Quota Exceeded" in response.json()["detail"]
