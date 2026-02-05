from llm_gateway.core.config import settings
from llm_gateway.core.interfaces import BaseLLMProvider, BaseRouter
from llm_gateway.schemas.chat import ChatRequest, ChatResponse


class SimpleRouter(BaseRouter):
    def __init__(self, providers: dict[str, BaseLLMProvider]):
        self.providers = providers
        self.default_provider = settings.DEFAULT_PROVIDER

    def _select_provider(self, model: str) -> BaseLLMProvider:
        # 1. 특정 모델명이 명시된 경우 (강제 적용)
        if model.startswith(("gpt", "o1", "text-embedding-3")):
            return self.providers["openai"]
        if model.startswith("gemini"):
            return self.providers["google"]

        # 2. 모델명이 'default'이거나 특정되지 않은 경우
        if model in ["default", "openai", "google"]:
            provider_key = (
                "openai"
                if model == "openai"
                else "google"
                if model == "google"
                else self.default_provider
            )
            return self.providers[provider_key]

        # 3. 그 외 기본 설정된 공급자 사용
        return self.providers[self.default_provider]

    async def route_chat(self, request: ChatRequest) -> ChatResponse:
        provider = self._select_provider(request.model)
        return await provider.chat_complete(request)

    def set_default_provider(self, provider: str):
        if provider not in self.providers:
            raise ValueError(f"Provider {provider} not available.")
        self.default_provider = provider
