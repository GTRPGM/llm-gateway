from llm_gateway.core.config import settings
from llm_gateway.schemas.chat import ChatRequest, ChatResponse
from llm_gateway.services.providers.base import BaseLLMProvider
from llm_gateway.services.providers.gemini import GeminiProvider


class LLMRouter:
    def __init__(self):
        self.providers: dict[str, BaseLLMProvider] = {}

        # Initialize Providers if keys are present
        if settings.GEMINI_API_KEY:
            self.providers["google"] = GeminiProvider()

        # 추후 OpenAI 등 추가

    def _get_provider_for_model(self, model: str) -> BaseLLMProvider:
        """
        Determines which provider to use based on the model name.
        """
        if "gemini" in model:
            if "google" not in self.providers:
                raise ValueError("Google provider is not configured (missing API key).")
            return self.providers["google"]

        # Default fallback or error
        # 현재는 Gemini만 있으므로 에러 처리하거나 기본값 설정
        if "google" in self.providers:
            return self.providers["google"]

        raise ValueError(f"No provider found for model: {model}")

    async def route_chat_completion(self, request: ChatRequest) -> ChatResponse:
        provider = self._get_provider_for_model(request.model)
        return await provider.chat_complete(request)


# Global Router Instance
llm_router = LLMRouter()
