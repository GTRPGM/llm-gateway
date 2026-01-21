from fastapi import FastAPI

from llm_gateway.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME, openapi_url=f"{settings.API_V1_STR}/openapi.json"
)


@app.get("/health", tags=["health"])
async def health_check():
    return {"status": "ok"}


@app.get("/", tags=["health"])
async def root():
    return {"message": "Welcome to LLM Gateway"}
