from unittest.mock import AsyncMock, patch

import pytest

from llm_gateway.schemas.chat import (
    ChatMessage,
    ChatResponse,
    ChatResponseChoice,
    ChatResponseChoiceDelta,
    ChatResponseChunk,
    ChatResponseChunkChoice,
)


@pytest.fixture
def mock_engine(app_instance):
    engine = app_instance.state.engine
    with patch.object(engine, "chat", new_callable=AsyncMock) as mock:
        yield mock


def test_chat_completions_streaming_success(mock_engine, client_instance):
    # Setup Mock Streaming Response
    mock_chunk = ChatResponseChunk(
        id="test-id",
        created=1234567890,
        model="gpt-4o",
        choices=[
            ChatResponseChunkChoice(
                index=0,
                delta=ChatResponseChoiceDelta(content="Hello"),
                finish_reason=None,
            )
        ],
    )

    async def mock_async_iterator():
        yield mock_chunk

    mock_engine.return_value = mock_async_iterator()

    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
    }

    # Execute
    response = client_instance.post("/api/v1/chat/completions", json=payload)

    # Verify
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    # Check body content
    body = response.text
    assert "data: " in body
    assert '"content":"Hello"' in body
    assert "data: [DONE]" in body


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


def test_chat_completions_internal_error(mock_engine, client_instance):
    mock_engine.side_effect = Exception("Unexpected error")

    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
    }

    response = client_instance.post("/api/v1/chat/completions", json=payload)

    assert response.status_code == 500
    assert response.json()["detail"] == "Internal Server Error"
