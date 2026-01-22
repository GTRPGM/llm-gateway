from llm_gateway.core.interfaces import BaseLLMProvider, BaseRouter
from llm_gateway.schemas.chat import ChatRequest, ChatResponse


class SimpleRouter(BaseRouter):
    def __init__(self, providers: dict[str, BaseLLMProvider]):
        self.providers = providers

    def _select_provider(self, model: str) -> BaseLLMProvider:
        if model.startswith("gemini"):
            return self.providers["google"]

        raise ValueError(f"Unsupported model: {model}")

    async def route_chat(self, request: ChatRequest) -> ChatResponse:
        provider = self._select_provider(request.model)
        return await provider.chat_complete(request)
