import logging
import traceback

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from llm_gateway.api.v1 import chat, gateway
from llm_gateway.core.config import settings
from llm_gateway.core.engine import LLMEngine
from llm_gateway.extensions.providers import GeminiProvider, OpenAIProvider
from llm_gateway.extensions.routers import SimpleRouter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def app() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_NAME,
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
    )

    providers = {
        "google": GeminiProvider(),
        "openai": OpenAIProvider(),
    }

    router = SimpleRouter(providers)
    engine = LLMEngine(router)

    app.state.engine = engine
    app.state.router = router  # Allow direct access to router for config

    app.include_router(chat.router, prefix=f"{settings.API_V1_STR}/chat", tags=["chat"])
    app.include_router(
        gateway.router, prefix=f"{settings.API_V1_STR}/gateway", tags=["gateway"]
    )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):
        body = await request.body()
        err_msg = (
            f"Validation Error: {exc.errors()} | "
            f"URL: {request.url} | Body: {body.decode()}"
        )
        logger.error(err_msg)
        return JSONResponse(
            status_code=422,
            content={"detail": exc.errors(), "body": body.decode()},
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        logger.error(f"HTTP {exc.status_code} Error: {exc.detail} | URL: {request.url}")
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        # Exception type check to avoid double logging if it's already an HTTPException
        if isinstance(exc, HTTPException):
            return await http_exception_handler(request, exc)

        logger.error(f"Unhandled Exception: {str(exc)}\n{traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal Server Error", "message": str(exc)},
        )

    @app.get("/")
    def root():
        return {"message": "LLM Gateway is running"}

    @app.get("/health")
    def health_check():
        return {"status": "ok"}

    return app
