from fastapi import FastAPI

from llm_gateway.api.v1 import chat
from llm_gateway.core.config import settings
from llm_gateway.core.engine import LLMEngine
from llm_gateway.extensions.providers import GeminiProvider
from llm_gateway.extensions.routers import SimpleRouter


def app() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_NAME,
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
    )

    providers = {"google": GeminiProvider()}

    router = SimpleRouter(providers)
    engine = LLMEngine(router)

    app.state.engine = engine  # 여기 중요

    app.include_router(chat.router, prefix=f"{settings.API_V1_STR}/chat", tags=["chat"])

    @app.get("/")
    def root():
        return {"message": "LLM Gateway is running"}

    @app.get("/health")
    def health_check():
        return {"status": "ok"}

    return app
