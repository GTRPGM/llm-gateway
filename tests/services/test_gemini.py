from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_gateway.schemas.chat import ChatMessage, ChatRequest
from llm_gateway.services.providers.gemini import GeminiProvider


@pytest.fixture
def mock_settings():
    with patch("llm_gateway.services.providers.gemini.settings") as mock:
        mock.GOOGLE_API_KEY = "fake-key"
        yield mock


@pytest.fixture
def mock_genai_client():
    with patch("llm_gateway.services.providers.gemini.genai.Client") as mock:
        yield mock


@pytest.mark.asyncio
async def test_gemini_chat_complete(mock_settings, mock_genai_client):
    # Setup Mock
    mock_client_instance = MagicMock()
    mock_chat_session = MagicMock()
    mock_response = MagicMock()

    mock_genai_client.return_value = mock_client_instance

    # client.chats.create -> returns chat session
    mock_client_instance.chats.create.return_value = mock_chat_session

    # chat.send_message is async
    mock_chat_session.send_message = AsyncMock(return_value=mock_response)

    # Response attribute
    mock_response.text = "Hello! I am Gemini 2.0."

    # Initialize Provider
    provider = GeminiProvider()

    # Create Request
    request = ChatRequest(
        model="gemini-2.0-flash",
        messages=[
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="Hi there!"),
        ],
    )

    # Execute
    response = await provider.chat_complete(request)

    # Verify
    assert response.choices[0].message.content == "Hello! I am Gemini 2.0."
    assert response.model == "gemini-2.0-flash"

    # Verify Calls
    mock_client_instance.chats.create.assert_called_once()

    # Check if system instruction was passed in config
    _, kwargs = mock_client_instance.chats.create.call_args
    assert kwargs["config"].system_instruction == "You are a helpful assistant."

    # Check send_message called with last user message
    mock_chat_session.send_message.assert_awaited_once_with(message="Hi there!")
