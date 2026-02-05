from typing import Any

from openai import AsyncOpenAI

from llm_gateway.core.config import settings
from llm_gateway.core.interfaces import BaseLLMProvider
from llm_gateway.schemas.chat import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatResponseChoice,
)


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

    async def chat_complete(self, request: ChatRequest) -> ChatResponse:
        if not self.client:
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is not set.")
            self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

        model_name = request.model
        if model_name in ["openai", "gpt"]:
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
            kwargs["response_format"] = request.response_format

        if request.tools:
            kwargs["tools"] = request.tools
            if request.tool_choice:
                kwargs["tool_choice"] = request.tool_choice

        # Call OpenAI API
        response = await self.client.chat.completions.create(**kwargs)

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
