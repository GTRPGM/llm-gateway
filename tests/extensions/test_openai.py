from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_gateway.core.config import settings
from llm_gateway.extensions.providers.openai import OpenAIProvider
from llm_gateway.schemas.chat import ChatMessage, ChatRequest, ChatResponseChunk


@pytest.mark.asyncio
async def test_openai_chat_complete_streaming(mock_openai_client):
    # Setup mock streaming response
    mock_chunk = MagicMock()
    mock_chunk.id = "chatcmpl-stream"
    mock_chunk.created = 123456789
    mock_chunk.model = "gpt-4o"

    mock_choice = MagicMock()
    mock_choice.index = 0
    mock_choice.delta.role = "assistant"
    mock_choice.delta.content = "Hello"
    mock_choice.delta.tool_calls = None
    mock_choice.finish_reason = None
    mock_chunk.choices = [mock_choice]

    async def mock_async_iterator():
        yield mock_chunk

    mock_openai_client.chat.completions.create.return_value = mock_async_iterator()

    with patch.object(settings, "OPENAI_API_KEY", "test-key"):
        provider = OpenAIProvider()
        request = ChatRequest(
            model="gpt-4o",
            messages=[ChatMessage(role="user", content="hello")],
            stream=True,
        )

        response = await provider.chat_complete(request)
        assert isinstance(response, AsyncIterator)

        chunks = []
        async for chunk in response:
            chunks.append(chunk)

        assert len(chunks) == 1
        assert isinstance(chunks[0], ChatResponseChunk)
        assert chunks[0].choices[0].delta.content == "Hello"


@pytest.fixture
def mock_openai_client():
    with patch("llm_gateway.extensions.providers.openai.AsyncOpenAI") as mock:
        client_instance = mock.return_value
        client_instance.chat = MagicMock()
        client_instance.chat.completions = MagicMock()
        client_instance.chat.completions.create = AsyncMock()
        yield client_instance


@pytest.mark.asyncio
async def test_openai_chat_complete_basic(mock_openai_client):
    # Setup mock response
    mock_choice = MagicMock()
    mock_choice.index = 0
    mock_choice.message.role = "assistant"
    mock_choice.message.content = "Hello from OpenAI"
    mock_choice.message.tool_calls = None
    mock_choice.finish_reason = "stop"

    mock_response = MagicMock()
    mock_response.id = "chatcmpl-123"
    mock_response.created = 123456789
    mock_response.model = "gpt-4o-mini"
    mock_response.choices = [mock_choice]

    mock_openai_client.chat.completions.create.return_value = mock_response

    with patch.object(settings, "OPENAI_API_KEY", "test-key"):
        provider = OpenAIProvider()
        request = ChatRequest(
            model="gpt-4o-mini", messages=[ChatMessage(role="user", content="hello")]
        )

        response = await provider.chat_complete(request)

        assert response.choices[0].message.content == "Hello from OpenAI"
        assert response.model == "gpt-4o-mini"
        mock_openai_client.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_openai_chat_complete_with_tools(mock_openai_client):
    # Setup mock tool response
    mock_tool_call = MagicMock()
    mock_tool_call.id = "call_123"
    mock_tool_call.type = "function"
    mock_tool_call.function.name = "get_weather"
    mock_tool_call.function.arguments = '{"location": "Seoul"}'

    mock_choice = MagicMock()
    mock_choice.index = 0
    mock_choice.message.role = "assistant"
    mock_choice.message.content = None
    mock_choice.message.tool_calls = [mock_tool_call]
    mock_choice.finish_reason = "tool_calls"

    mock_response = MagicMock()
    mock_response.id = "chatcmpl-tool"
    mock_response.created = 123456789
    mock_response.model = "gpt-4o"
    mock_response.choices = [mock_choice]

    mock_openai_client.chat.completions.create.return_value = mock_response

    with patch.object(settings, "OPENAI_API_KEY", "test-key"):
        provider = OpenAIProvider()
        request = ChatRequest(
            model="gpt-4o",
            messages=[
                ChatMessage(role="system", content="You are a helpful assistant"),
                ChatMessage(role="user", content="What's the weather?"),
                ChatMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": "{}"},
                        }
                    ],
                ),
                ChatMessage(role="tool", content="Sunny", tool_call_id="call_123"),
            ],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
        )

        response = await provider.chat_complete(request)

        assert response.choices[0].finish_reason == "tool_calls"
        assert (
            response.choices[0].message.tool_calls[0]["function"]["name"]
            == "get_weather"
        )


@pytest.mark.asyncio
async def test_openai_chat_complete_json_mode(mock_openai_client):
    mock_response = MagicMock()
    mock_response.id = "chatcmpl-json"
    mock_response.created = 123456789
    mock_response.model = "gpt-4o"
    mock_response.choices = []

    mock_openai_client.chat.completions.create.return_value = mock_response

    with patch.object(settings, "OPENAI_API_KEY", "test-key"):
        provider = OpenAIProvider()
        request = ChatRequest(
            model="gpt-4o",
            messages=[ChatMessage(role="user", content="Output JSON")],
            response_format={"type": "json_object"},
        )

        await provider.chat_complete(request)

        _, kwargs = mock_openai_client.chat.completions.create.call_args
        assert kwargs["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_openai_chat_complete_json_schema_missing_name_is_filled(
    mock_openai_client,
):
    mock_response = MagicMock()
    mock_response.id = "chatcmpl-jsonschema"
    mock_response.created = 123456789
    mock_response.model = "gpt-4o"
    mock_response.choices = []

    mock_openai_client.chat.completions.create.return_value = mock_response

    with patch.object(settings, "OPENAI_API_KEY", "test-key"):
        provider = OpenAIProvider()
        request = ChatRequest(
            model="gpt-4o",
            messages=[ChatMessage(role="user", content="Output JSON schema")],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    # name intentionally omitted
                    "schema": {
                        "type": "object",
                        "properties": {"ok": {"type": "boolean"}},
                        "required": ["ok"],
                    }
                },
            },
        )

        await provider.chat_complete(request)

        _, kwargs = mock_openai_client.chat.completions.create.call_args
        rf = kwargs["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["name"] == "structured_output"


@pytest.mark.asyncio
async def test_openai_chat_complete_no_api_key():
    with patch.object(settings, "OPENAI_API_KEY", None):
        provider = OpenAIProvider()
        # Reset client if it was initialized in previous tests
        provider.client = None
        request = ChatRequest(model="gpt-4o-mini", messages=[])
        with pytest.raises(ValueError, match="OPENAI_API_KEY is not set"):
            await provider.chat_complete(request)
