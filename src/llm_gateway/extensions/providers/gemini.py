import json
import logging
import time
import uuid
from typing import AsyncIterator, Union

from fastapi import HTTPException
from google import genai
from google.genai import types

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


class GeminiProvider(BaseLLMProvider):
    def __init__(self):
        if not settings.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is not set in environment variables.")
        # Client 초기화는 동기적으로 수행
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
            elif msg.role == "tool":
                # Tool Response
                history.append(
                    types.Content(
                        role="tool",
                        parts=[
                            types.Part(
                                function_response=types.FunctionResponse(
                                    name=msg.tool_call_id,
                                    response={
                                        "result": msg.content
                                    },  # content is usually JSON string
                                )
                            )
                        ],
                    )
                )
            else:
                role = "model" if msg.role == "assistant" else "user"
                parts = []

                if msg.content:
                    parts.append(types.Part(text=msg.content))

                if msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        if tool_call.get("type") == "function":
                            fn = tool_call["function"]
                            parts.append(
                                types.Part(
                                    function_call=types.FunctionCall(
                                        name=fn["name"],
                                        args=json.loads(fn["arguments"])
                                        if isinstance(fn["arguments"], str)
                                        else fn["arguments"],
                                    )
                                )
                            )

                history.append(types.Content(role=role, parts=parts))

        return history, system_instruction

    def _convert_tools(self, tools: list[dict] | None) -> list[types.Tool] | None:
        if not tools:
            return None

        function_declarations = []
        for tool in tools:
            if tool.get("type") == "function":
                fn = tool["function"]
                # OpenAI schema to Gemini Schema mapping
                function_declarations.append(
                    types.FunctionDeclaration(
                        name=fn.get("name"),
                        description=fn.get("description"),
                        parameters=fn.get("parameters"),
                    )
                )

        if not function_declarations:
            return None

        return [types.Tool(function_declarations=function_declarations)]

    async def chat_complete(
        self, request: ChatRequest
    ) -> Union[ChatResponse, AsyncIterator[ChatResponseChunk]]:
        # 모델명 결정
        model_name = request.model
        if model_name in [None, "", "gemini", "google", "default"]:
            model_name = settings.GEMINI_DEFAULT_MODEL

        history, system_instruction = self._convert_messages(request.messages)

        # Response Format Handling (JSON Mode)
        response_mime_type = "text/plain"
        response_schema = None

        if request.response_format:
            if request.response_format.get("type") == "json_object":
                response_mime_type = "application/json"
            elif request.response_format.get("type") == "json_schema":
                response_mime_type = "application/json"
                if "json_schema" in request.response_format:
                    schema_data = request.response_format["json_schema"].get("schema")
                    if schema_data:
                        response_schema = schema_data

        # Tools Handling
        gemini_tools = self._convert_tools(request.tools)

        tool_config = None
        if request.tool_choice:
            mode = "AUTO"
            allowed_function_names = None

            if isinstance(request.tool_choice, str):
                if request.tool_choice == "none":
                    mode = "NONE"
                elif request.tool_choice == "auto":
                    mode = "AUTO"
                elif request.tool_choice == "required":
                    mode = "ANY"
            elif isinstance(request.tool_choice, dict):
                if request.tool_choice.get("type") == "function":
                    mode = "ANY"
                    fn_name = request.tool_choice.get("function", {}).get("name")
                    if fn_name:
                        allowed_function_names = [fn_name]

            tool_config = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode=mode, allowed_function_names=allowed_function_names
                )
            )

        config = types.GenerateContentConfig(
            temperature=request.temperature,
            max_output_tokens=request.max_tokens,
            system_instruction=system_instruction,
            response_mime_type=response_mime_type,
            response_schema=response_schema,
            tools=gemini_tools,
            tool_config=tool_config,
        )

        chat = self.client.aio.chats.create(
            model=model_name,
            config=config,
            history=history[:-1] if history and history[-1].role == "user" else history,
        )

        last_message_content = ""
        if history and history[-1].role == "user":
            parts = history[-1].parts
            for part in parts:
                if part.text:
                    last_message_content = part.text
                    break
        else:
            last_message_content = "..."

        if request.stream:
            try:
                stream_response = await chat.send_message_stream(
                    message=last_message_content
                )
            except Exception as e:
                logger.error(f"Gemini API Stream Error: {str(e)}")
                raise HTTPException(
                    status_code=502, detail=f"Error from Gemini Provider: {str(e)}"
                ) from e

            request_id = f"chatcmpl-{uuid.uuid4()}"

            async def gen():
                try:
                    async for chunk in stream_response:
                        content = ""
                        tool_calls = []
                        finish_reason = None

                        if chunk.candidates and chunk.candidates[0].content.parts:
                            for part in chunk.candidates[0].content.parts:
                                if part.text:
                                    content += part.text
                                if part.function_call:
                                    tool_calls.append(
                                        {
                                            "id": part.function_call.name,
                                            "type": "function",
                                            "function": {
                                                "name": part.function_call.name,
                                                "arguments": json.dumps(
                                                    part.function_call.args
                                                ),
                                            },
                                        }
                                    )

                            if chunk.candidates[0].finish_reason:
                                # Mapping Gemini finish reason to OpenAI
                                fr = chunk.candidates[0].finish_reason
                                if fr == "STOP":
                                    finish_reason = "stop"
                                elif fr == "MAX_TOKENS":
                                    finish_reason = "length"
                                elif fr == "SAFETY":
                                    finish_reason = "content_filter"
                                else:
                                    finish_reason = str(fr).lower()

                        yield ChatResponseChunk(
                            id=request_id,
                            created=int(time.time()),
                            model=model_name,
                            choices=[
                                ChatResponseChunkChoice(
                                    index=0,
                                    delta=ChatResponseChoiceDelta(
                                        content=content if content else None,
                                        tool_calls=tool_calls if tool_calls else None,
                                    ),
                                    finish_reason=finish_reason,
                                )
                            ],
                        )
                except Exception as e:
                    logger.error(f"Gemini Stream Chunk Error: {str(e)}")

            return gen()

        # 비동기 호출
        try:
            response = await chat.send_message(message=last_message_content)
        except Exception as e:
            logger.error(f"Gemini API Error: {str(e)}")
            raise HTTPException(
                status_code=502, detail=f"Error from Gemini Provider: {str(e)}"
            ) from e

        # Response parsing
        response_content = None
        tool_calls = []

        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.text:
                    if response_content is None:
                        response_content = ""
                    response_content += part.text

                if part.function_call:
                    tool_calls.append(
                        {
                            "id": part.function_call.name,
                            "type": "function",
                            "function": {
                                "name": part.function_call.name,
                                "arguments": json.dumps(part.function_call.args),
                            },
                        }
                    )

        return ChatResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            created=int(time.time()),
            model=model_name,
            choices=[
                ChatResponseChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=response_content if response_content else "",
                        tool_calls=tool_calls if tool_calls else None,
                    ),
                    finish_reason="tool_calls" if tool_calls else "stop",
                )
            ],
        )
