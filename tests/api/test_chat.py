from unittest.mock import AsyncMock, patch

import pytest

from llm_gateway.schemas.chat import ChatMessage, ChatResponse, ChatResponseChoice


@pytest.fixture
def mock_engine(app_instance):
    engine = app_instance.state.engine
    with patch.object(engine, "chat", new_callable=AsyncMock) as mock:
        yield mock


def test_chat_completions_success(mock_engine, client_instance):
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
    mock_engine.return_value = mock_response

    # Request Payload
    payload = {
        "model": "gemini-1.5-flash",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    # Execute
    response = client_instance.post("/api/v1/chat/completions", json=payload)

    # Verify
    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["message"]["content"] == "Hello via API"
    assert data["model"] == "gemini-1.5-flash"


def test_chat_completions_provider_error(mock_engine, client_instance):
    # Setup Mock to Raise Error
    mock_engine.side_effect = ValueError("Invalid model")

    payload = {
        "model": "unknown-model",
        "messages": [{"role": "user", "content": "Hi"}],
    }

    response = client_instance.post("/api/v1/chat/completions", json=payload)

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid model"
