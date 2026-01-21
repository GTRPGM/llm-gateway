from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "LLM Gateway"
    API_V1_STR: str = "/api/v1"

    # LLM Keys
    OPENAI_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None

    # Model Configuration
    GEMINI_DEFAULT_MODEL: str = "gemini-2.0-flash-lite-001"

    # Observability
    LANGSMITH_TRACING: bool = False
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGSMITH_API_KEY: str | None = None
    LANGSMITH_PROJECT: str = "llm-gateway"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=True
    )


settings = Settings()
