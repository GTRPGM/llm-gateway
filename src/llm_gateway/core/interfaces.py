from abc import ABC, abstractmethod
from typing import AsyncIterator, Union

from llm_gateway.schemas.chat import ChatRequest, ChatResponse, ChatResponseChunk


class BaseLLMProvider(ABC):
    """
    Abstract base class for all LLM providers (e.g., OpenAI, Gemini).
    """

    @abstractmethod
    async def chat_complete(
        self, request: ChatRequest
    ) -> Union[ChatResponse, AsyncIterator[ChatResponseChunk]]:
        """
        Generates a response from the LLM based on the chat history.
        """
        pass


class BaseRouter(ABC):
    @abstractmethod
    async def route_chat(
        self, request: ChatRequest
    ) -> Union[ChatResponse, AsyncIterator[ChatResponseChunk]]:
        raise NotImplementedError
