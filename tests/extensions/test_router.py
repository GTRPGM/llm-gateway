from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_gateway.extensions.routers.simple_router import SimpleRouter
from llm_gateway.schemas.chat import ChatMessage, ChatRequest


@pytest.fixture
def mock_providers():
    return {"google": MagicMock(), "openai": MagicMock()}


@pytest.fixture
def router(mock_providers):
    return SimpleRouter(mock_providers)


@pytest.mark.asyncio
async def test_router_select_openai_by_model_name(router, mock_providers):
    request = ChatRequest(
        model="gpt-4o", messages=[ChatMessage(role="user", content="hi")]
    )
    mock_providers["openai"].chat_complete = AsyncMock()

    await router.route_chat(request)
    mock_providers["openai"].chat_complete.assert_called_once()
    mock_providers["google"].chat_complete.assert_not_called()


@pytest.mark.asyncio
async def test_router_select_google_by_model_name(router, mock_providers):
    request = ChatRequest(
        model="gemini-pro", messages=[ChatMessage(role="user", content="hi")]
    )
    mock_providers["google"].chat_complete = AsyncMock()

    await router.route_chat(request)
    mock_providers["google"].chat_complete.assert_called_once()
    mock_providers["openai"].chat_complete.assert_not_called()


@pytest.mark.asyncio
async def test_router_default_provider(router, mock_providers):
    router.set_default_provider("openai")
    request = ChatRequest(
        model="default", messages=[ChatMessage(role="user", content="hi")]
    )
    mock_providers["openai"].chat_complete = AsyncMock()

    await router.route_chat(request)
    mock_providers["openai"].chat_complete.assert_called_once()

    router.set_default_provider("google")
    mock_providers["google"].chat_complete = AsyncMock()
    await router.route_chat(request)
    mock_providers["google"].chat_complete.assert_called_once()


def test_router_set_invalid_provider(router):
    with pytest.raises(ValueError, match="Provider invalid not available"):
        router.set_default_provider("invalid")
