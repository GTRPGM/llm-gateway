import asyncio
import logging
from typing import Any, AsyncIterator, Union

from fastapi import HTTPException
from openai import AsyncOpenAI, OpenAIError

from llm_gateway.core.config import settings
from llm_gateway.core.interfaces import BaseLLMProvider
from llm_gateway.schemas.chat import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatResponseChoice,
    ChatResponseChoiceDelta,
    ChatResponseChunk,
    ChatResponseChunkChoice,
)

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLMProvider):
    def __init__(self):
        self.client = None
        if settings.OPENAI_API_KEY:
            self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    def _convert_messages(self, messages: list[ChatMessage]) -> list[dict[str, Any]]:
        """
        Convert standard ChatMessage to OpenAI message format.
        """
        openai_messages = []
        for msg in messages:
            m = {"role": msg.role, "content": msg.content}
            if msg.tool_calls:
                m["tool_calls"] = msg.tool_calls
            if msg.tool_call_id:
                m["tool_call_id"] = msg.tool_call_id
            openai_messages.append(m)
        return openai_messages

    async def chat_complete(
        self, request: ChatRequest
    ) -> Union[ChatResponse, AsyncIterator[ChatResponseChunk]]:
        if not self.client:
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is not set.")
            self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

        model_name = request.model
        if model_name in ["openai", "gpt", "default"]:
            model_name = settings.OPENAI_DEFAULT_MODEL

        # Prepare arguments for OpenAI API
        kwargs = {
            "model": model_name,
            "messages": self._convert_messages(request.messages),
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": request.stream,
        }

        if request.response_format:
            # OpenAI requires response_format.json_schema.name when using json_schema.
            # Some upstream callers only provide
            # {"type":"json_schema","json_schema":{"schema":...}}.
            rf = dict(request.response_format)
            if rf.get("type") == "json_schema":
                js = rf.get("json_schema")
                if isinstance(js, dict):
                    if not js.get("name"):
                        js = dict(js)
                        js["name"] = "structured_output"
                        rf["json_schema"] = js
            kwargs["response_format"] = rf

        if request.tools:
            kwargs["tools"] = request.tools
            if request.tool_choice:
                kwargs["tool_choice"] = request.tool_choice

        # Call OpenAI API with retry on transient provider failures.
        response = None
        last_error: Exception | None = None
        for attempt in range(1, settings.OPENAI_RETRY_ATTEMPTS + 1):
            try:
                response = await self.client.chat.completions.create(**kwargs)
                break
            except OpenAIError as e:
                last_error = e
                logger.warning(
                    "OpenAI API Error (attempt %s/%s): %s",
                    attempt,
                    settings.OPENAI_RETRY_ATTEMPTS,
                    str(e),
                )
                if attempt >= settings.OPENAI_RETRY_ATTEMPTS:
                    raise HTTPException(
                        status_code=502, detail=f"Error from OpenAI Provider: {str(e)}"
                    ) from e
                await asyncio.sleep(
                    settings.OPENAI_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                )
            except Exception as e:
                last_error = e
                logger.warning(
                    "Unexpected OpenAI provider error (attempt %s/%s): %s",
                    attempt,
                    settings.OPENAI_RETRY_ATTEMPTS,
                    str(e),
                )
                if attempt >= settings.OPENAI_RETRY_ATTEMPTS:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Unexpected Error in OpenAIProvider: {str(e)}",
                    ) from e
                await asyncio.sleep(
                    settings.OPENAI_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                )

        if response is None:
            detail = str(last_error) if last_error else "Unknown OpenAI failure"
            raise HTTPException(status_code=500, detail=detail)

        if request.stream:

            async def gen():
                try:
                    async for chunk in response:
                        choices = []
                        for choice in chunk.choices:
                            delta = ChatResponseChoiceDelta(
                                role=choice.delta.role,
                                content=choice.delta.content,
                            )
                            if choice.delta.tool_calls:
                                delta.tool_calls = [
                                    {
                                        "index": tc.index,
                                        "id": tc.id,
                                        "type": tc.type,
                                        "function": {
                                            "name": tc.function.name,
                                            "arguments": tc.function.arguments,
                                        },
                                    }
                                    for tc in choice.delta.tool_calls
                                ]

                            choices.append(
                                ChatResponseChunkChoice(
                                    index=choice.index,
                                    delta=delta,
                                    finish_reason=choice.finish_reason,
                                )
                            )

                        yield ChatResponseChunk(
                            id=chunk.id,
                            created=chunk.created,
                            model=chunk.model,
                            choices=choices,
                        )
                except OpenAIError as e:
                    logger.error(f"OpenAI Stream Error: {str(e)}")
                    # Streaming errors are harder to propagate as HTTP exceptions
                    # but we log them for visibility.
                except Exception as e:
                    logger.error(f"Unexpected OpenAI Stream Error: {str(e)}")

            return gen()

        # Convert OpenAI response to our ChatResponse
        choices = []
        for choice in response.choices:
            tool_calls = None
            if choice.message.tool_calls:
                tool_calls = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in choice.message.tool_calls
                ]

            choices.append(
                ChatResponseChoice(
                    index=choice.index,
                    message=ChatMessage(
                        role="assistant",
                        content=choice.message.content or "",
                        tool_calls=tool_calls,
                    ),
                    finish_reason=choice.finish_reason,
                )
            )

        return ChatResponse(
            id=response.id,
            created=response.created,
            model=response.model,
            choices=choices,
        )
