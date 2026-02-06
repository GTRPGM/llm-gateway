import json
import os
from typing import Any, Callable, List, Optional, Sequence, Union

import httpx
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, Field

# Configuration defaults
REMOTE_HOST = os.getenv("REMOTE_HOST", "localhost")
DEFAULT_LLM_GATEWAY_URL = f"http://{REMOTE_HOST}:8060"
DEFAULT_MODEL_NAME = "default"


class GatewayChatModel(BaseChatModel):
    """
    Custom LangChain ChatModel adapter for the LLM Gateway.
    Completely independent of internal project schemas.
    """

    base_url: str = Field(default=DEFAULT_LLM_GATEWAY_URL)
    model_name: str = Field(default=DEFAULT_MODEL_NAME)

    # Use a long timeout for LLM generation
    timeout: float = 600.0

    @property
    def _llm_type(self) -> str:
        return "llm_gateway"

    def _convert_message_to_dict(self, message: BaseMessage) -> dict:
        """Converts LangChain message to a plain dictionary matching the Gateway API."""
        role = "user"
        tool_calls = None
        tool_call_id = None

        if isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, AIMessage):
            role = "assistant"
            if message.tool_calls:
                tool_calls = [
                    {
                        "id": tc.get("id"),
                        "type": "function",
                        "function": {
                            "name": tc.get("name"),
                            "arguments": json.dumps(tc.get("args")),
                        },
                    }
                    for tc in message.tool_calls
                ]
        elif isinstance(message, ToolMessage):
            role = "tool"
            tool_call_id = message.tool_call_id
        elif isinstance(message, ChatMessage):
            role = message.role

        return {
            "role": role,
            "content": str(message.content) if message.content else None,
            "tool_calls": tool_calls,
            "tool_call_id": tool_call_id,
        }

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError(
            "Sync generation not implemented. Use ainvoke/agenerate."
        )

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Build request body using plain dicts
        schema_messages = [self._convert_message_to_dict(m) for m in messages]

        request_body = {
            "model": self.model_name,
            "messages": schema_messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens"),
            "response_format": kwargs.get("response_format"),
            "tools": kwargs.get("tools"),
            "tool_choice": kwargs.get("tool_choice"),
        }

        # Remove None values
        request_body = {k: v for k, v in request_body.items() if v is not None}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            target_url = f"{self.base_url.rstrip('/')}/api/v1/chat/completions"

            try:
                response = await client.post(target_url, json=request_body)
                response.raise_for_status()
            except httpx.RequestError as exc:
                print(f"❌ Connection error: {exc}")
                raise
            except httpx.HTTPStatusError as exc:
                print(f"❌ HTTP error {exc.response.status_code}: {exc.response.text}")
                raise

            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                return ChatResult(generations=[])

            choice = choices[0]
            choice_msg = choice.get("message", {})
            msg_kwargs = {}

            # Parse tool calls from response
            tool_calls_data = choice_msg.get("tool_calls")
            if tool_calls_data:
                lc_tool_calls = []
                for tc in tool_calls_data:
                    fn = tc.get("function", {})
                    args_str = fn.get("arguments", "{}")
                    try:
                        args = json.loads(args_str)
                    except json.JSONDecodeError:
                        args = {}

                    lc_tool_calls.append(
                        {"name": fn.get("name"), "args": args, "id": tc.get("id")}
                    )
                msg_kwargs["tool_calls"] = lc_tool_calls

            raw_content = choice_msg.get("content") or ""
            parsed_content = None
            response_format = kwargs.get("response_format")

            # Structured Output handling
            if response_format and isinstance(raw_content, str):
                fmt_type = response_format.get("type")
                if fmt_type in ("json_object", "json_schema"):
                    try:
                        parsed_content = json.loads(raw_content)
                    except json.JSONDecodeError:
                        parsed_content = None

            final_content: Any
            if parsed_content is not None:
                final_content = (
                    parsed_content
                    if isinstance(parsed_content, list)
                    else [parsed_content]
                )
                msg_kwargs["parsed"] = parsed_content
            else:
                final_content = raw_content

            generation = ChatGeneration(
                message=AIMessage(
                    content=final_content,
                    additional_kwargs=msg_kwargs,
                    tool_calls=msg_kwargs.get("tool_calls", []),
                ),
                generation_info={"finish_reason": choice.get("finish_reason")},
            )

            return ChatResult(generations=[generation])

    def bind_tools(
        self,
        tools: Sequence[Union[dict[str, Any], type[BaseModel], Callable, BaseTool]],
        **kwargs: Any,
    ) -> Runnable[Any, AIMessage]:
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        return self.bind(tools=formatted_tools, **kwargs)

    def with_structured_output(
        self,
        schema: Any,
        *,
        method: str = "json_object",
        **kwargs: Any,
    ) -> RunnableLambda:
        async def _call(messages: List[BaseMessage]) -> Any:
            schema_str = ""
            if hasattr(schema, "model_json_schema"):
                schema_str = json.dumps(
                    schema.model_json_schema(), indent=2, ensure_ascii=False
                )

            modified_messages = list(messages)
            schema_instruction = (
                "\n\nYour response MUST be a single JSON object "
                f"matching this schema:\n```json\n{schema_str}\n```\n"
                "Do not include any explanation or markdown outside the JSON."
            )

            system_msg_index = -1
            for i, m in enumerate(modified_messages):
                if isinstance(m, SystemMessage):
                    system_msg_index = i
                    break

            if system_msg_index != -1:
                modified_messages[system_msg_index] = SystemMessage(
                    content=modified_messages[system_msg_index].content
                    + schema_instruction
                )
            else:
                modified_messages.insert(0, SystemMessage(content=schema_instruction))

            result = await self.ainvoke(
                modified_messages,
                response_format={"type": "json_object"},
            )

            content = result.content
            if isinstance(content, list) and len(content) > 0:
                data = content[0]
            else:
                data = content

            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to parse JSON response: {data}") from e

            return schema.model_validate(data)

        return RunnableLambda(_call)

    async def check_health(self) -> bool:
        """Check if LLM Gateway is reachable."""
        async with httpx.AsyncClient(timeout=3.0) as client:
            try:
                resp = await client.get(f"{self.base_url.rstrip('/')}/health")
                return resp.status_code == 200
            except Exception:
                return False
