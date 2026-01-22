from abc import ABC, abstractmethod

from llm_gateway.schemas.chat import ChatRequest, ChatResponse


class BaseLLMProvider(ABC):
    """
    Abstract base class for all LLM providers (e.g., OpenAI, Gemini).
    """

    @abstractmethod
    async def chat_complete(self, request: ChatRequest) -> ChatResponse:
        """
        Generates a response from the LLM based on the chat history.
        """
        pass


class BaseRouter(ABC):
    @abstractmethod
    async def route_chat(self, request: ChatRequest) -> ChatResponse:
        raise NotImplementedError