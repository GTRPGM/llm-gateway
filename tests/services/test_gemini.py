import json
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
async def test_gemini_chat_complete_basic(mock_settings, mock_genai_client):
    # Setup Mock
    mock_client_instance = MagicMock()
    mock_chat_session = MagicMock()
    mock_response = MagicMock()

    mock_genai_client.return_value = mock_client_instance
    mock_client_instance.aio.chats.create.return_value = mock_chat_session
    mock_chat_session.send_message = AsyncMock(return_value=mock_response)

    # Mock Response Structure for Text
    mock_part = MagicMock()
    mock_part.text = "Hello! I am Gemini 2.0."
    mock_part.function_call = None

    mock_candidate = MagicMock()
    mock_candidate.content.parts = [mock_part]
    mock_response.candidates = [mock_candidate]

    # Initialize Provider
    provider = GeminiProvider()

    request = ChatRequest(
        model="gemini-2.0-flash",
        messages=[
            ChatMessage(role="system", content="System"),
            ChatMessage(role="user", content="Hi"),
        ],
    )

    response = await provider.chat_complete(request)

    assert response.choices[0].message.content == "Hello! I am Gemini 2.0."
    mock_chat_session.send_message.assert_awaited_once_with(message="Hi")


@pytest.mark.asyncio
async def test_gemini_chat_complete_json_mode(mock_settings, mock_genai_client):
    mock_client_instance = MagicMock()
    mock_chat_session = MagicMock()
    mock_response = MagicMock()

    mock_genai_client.return_value = mock_client_instance
    mock_client_instance.aio.chats.create.return_value = mock_chat_session
    mock_chat_session.send_message = AsyncMock(return_value=mock_response)

    # Mock Response
    mock_part = MagicMock()
    mock_part.text = '{"key": "value"}'
    mock_part.function_call = None
    mock_response.candidates = [MagicMock(content=MagicMock(parts=[mock_part]))]

    provider = GeminiProvider()

    request = ChatRequest(
        model="gemini-2.0-flash",
        messages=[ChatMessage(role="user", content="Output JSON")],
        response_format={"type": "json_object"},
    )

    await provider.chat_complete(request)

    # Verify Config
    _, kwargs = mock_client_instance.aio.chats.create.call_args
    assert kwargs["config"].response_mime_type == "application/json"


@pytest.mark.asyncio
async def test_gemini_chat_complete_tool_call(mock_settings, mock_genai_client):
    mock_client_instance = MagicMock()
    mock_chat_session = MagicMock()
    mock_response = MagicMock()

    mock_genai_client.return_value = mock_client_instance
    mock_client_instance.aio.chats.create.return_value = mock_chat_session
    mock_chat_session.send_message = AsyncMock(return_value=mock_response)

    # Mock Tool Call Response
    mock_part = MagicMock()
    mock_part.text = None
    mock_part.function_call.name = "get_weather"
    mock_part.function_call.args = {"location": "Seoul"}

    mock_response.candidates = [MagicMock(content=MagicMock(parts=[mock_part]))]

    provider = GeminiProvider()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            },
        }
    ]

    request = ChatRequest(
        model="gemini-2.0-flash",
        messages=[ChatMessage(role="user", content="Weather in Seoul?")],
        tools=tools,
    )

    response = await provider.chat_complete(request)

    # Verify Response
    assert response.choices[0].finish_reason == "tool_calls"
    tool_call = response.choices[0].message.tool_calls[0]
    assert tool_call["function"]["name"] == "get_weather"
    assert json.loads(tool_call["function"]["arguments"]) == {"location": "Seoul"}

    # Verify Config
    _, kwargs = mock_client_instance.aio.chats.create.call_args
    assert kwargs["config"].tools is not None
