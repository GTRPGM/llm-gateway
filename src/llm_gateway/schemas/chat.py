from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None  # Tool 호출 정보
    tool_call_id: str | None = None  # Tool 응답 시 해당 호출 ID

    model_config = {
        "json_schema_extra": {"example": {"role": "user", "content": "안녕하세요!"}}
    }


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int | None = None
    stream: bool = False

    # Structured Output 지원
    # 예: {"type": "json_object"} 또는 {"type": "json_schema", "json_schema": {...}}
    response_format: dict[str, Any] | None = None

    # Tool(Function) Calling 지원
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "안녕하세요! 간단한 인사를 해주세요."},
                ],
                "temperature": 0.7,
                "stream": False,
            }
        }
    }


class ChatResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str | None = None


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatResponseChoice]


class ChatResponseChoiceDelta(BaseModel):
    role: str | None = None
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ChatResponseChunkChoice(BaseModel):
    index: int
    delta: ChatResponseChoiceDelta
    finish_reason: str | None = None


class ChatResponseChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatResponseChunkChoice]
