from typing import AsyncIterator, Union

from llm_gateway.core.config import settings
from llm_gateway.core.interfaces import BaseLLMProvider, BaseRouter
from llm_gateway.schemas.chat import ChatRequest, ChatResponse, ChatResponseChunk


class SimpleRouter(BaseRouter):
    def __init__(self, providers: dict[str, BaseLLMProvider]):
        self.providers = providers
        self.default_provider = settings.DEFAULT_PROVIDER
        self.override_provider = None

    def _select_provider(self, model: str) -> BaseLLMProvider:
        # 0. Override Provider가 설정된 경우 최우선 적용
        if self.override_provider:
            return self.providers[self.override_provider]

        # 1. 특정 모델명이 명시된 경우 (강제 적용)
        if model.startswith(("gpt", "o1", "o3")):
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

    async def route_chat(
        self, request: ChatRequest
    ) -> Union[ChatResponse, AsyncIterator[ChatResponseChunk]]:
        # Override 설정 시, 모델명이 타 공급자용이면 'default'로 치환하여 에러 방지
        if self.override_provider:
            model = request.model
            is_openai_model = model.startswith(("gpt", "o1", "o3"))
            is_gemini_model = model.startswith("gemini")

            if self.override_provider == "openai" and is_gemini_model:
                request.model = "default"
            elif self.override_provider == "google" and is_openai_model:
                request.model = "default"

        provider = self._select_provider(request.model)
        return await provider.chat_complete(request)

    def set_default_provider(self, provider: str):
        if provider not in self.providers:
            raise ValueError(f"Provider {provider} not available.")
        self.default_provider = provider

    def set_override_provider(self, provider: str | None):
        if provider and provider not in self.providers:
            raise ValueError(f"Provider {provider} not available.")
        self.override_provider = provider
