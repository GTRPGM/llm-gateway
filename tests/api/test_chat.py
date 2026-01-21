from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from llm_gateway.main import app
from llm_gateway.schemas.chat import ChatMessage, ChatResponse, ChatResponseChoice

client = TestClient(app)


@pytest.fixture
def mock_router():
    with patch("llm_gateway.api.v1.chat.llm_router") as mock:
        yield mock


def test_chat_completions_success(mock_router):
    # Setup Mock Response
    mock_response = ChatResponse(
        id="test-id",
        created=1234567890,
        model="gemini-1.5-flash",
        choices=[
            ChatResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content="Hello via API"),
                finish_reason="stop",
            )
        ],
    )
    mock_router.route_chat_completion = AsyncMock(return_value=mock_response)

    # Request Payload
    payload = {
        "model": "gemini-1.5-flash",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    # Execute
    response = client.post("/api/v1/chat/completions", json=payload)

    # Verify
    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["message"]["content"] == "Hello via API"
    assert data["model"] == "gemini-1.5-flash"


def test_chat_completions_provider_error(mock_router):
    # Setup Mock to Raise Error
    mock_router.route_chat_completion = AsyncMock(
        side_effect=ValueError("Invalid model")
    )

    payload = {
        "model": "unknown-model",
        "messages": [{"role": "user", "content": "Hi"}],
    }

    response = client.post("/api/v1/chat/completions", json=payload)

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid model"
