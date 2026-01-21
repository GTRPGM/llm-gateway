import time
import uuid

from google import genai
from google.genai import types

from llm_gateway.core.config import settings
from llm_gateway.schemas.chat import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatResponseChoice,
)
from llm_gateway.services.providers.base import BaseLLMProvider


class GeminiProvider(BaseLLMProvider):
    def __init__(self):
        if not settings.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is not set in environment variables.")

        self.client = genai.Client(api_key=settings.GOOGLE_API_KEY)

    def _convert_messages(
        self, messages: list[ChatMessage]
    ) -> tuple[list[types.Content], str | None]:
        """
        Convert standard ChatMessage to Gemini Content format.
        """
        history = []
        system_instruction = None

        for msg in messages:
            if msg.role == "system":
                if system_instruction is None:
                    system_instruction = msg.content
                else:
                    system_instruction += f"\n{msg.content}"
            else:
                role = "model" if msg.role == "assistant" else "user"
                history.append(
                    types.Content(role=role, parts=[types.Part(text=msg.content)])
                )

        return history, system_instruction

    async def chat_complete(self, request: ChatRequest) -> ChatResponse:
        model_name = request.model if "gemini" in request.model else "gemini-1.5-flash"

        history, system_instruction = self._convert_messages(request.messages)
        config = types.GenerateContentConfig(
            temperature=request.temperature,
            max_output_tokens=request.max_tokens,
            system_instruction=system_instruction,
        )
        chat = self.client.chats.create(
            model=model_name,
            config=config,
            history=history[:-1] if history and history[-1].role == "user" else history,
        )

        last_message_content = ""
        if history and history[-1].role == "user":
            # Extract text from the last user message
            # history[-1].parts[0].text assuming text part
            last_message_content = history[-1].parts[0].text
        else:
            # Fallback or error case
            last_message_content = "..."

        # 비동기 호출
        response = await chat.send_message(message=last_message_content)

        return ChatResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            created=int(time.time()),
            model=model_name,
            choices=[
                ChatResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response.text),
                    finish_reason="stop",
                )
            ],
        )
