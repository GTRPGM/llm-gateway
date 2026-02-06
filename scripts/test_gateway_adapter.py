import asyncio
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Import the adapter
try:
    from scripts.gateway_adapter import GatewayChatModel
except ImportError:
    import sys

    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from gateway_adapter import GatewayChatModel


# --- Test Models & Tools ---


class CountryInfo(BaseModel):
    """Information about a country."""

    name: str = Field(description="Name of the country")
    capital: str = Field(description="Capital city of the country")
    population: int = Field(description="Approximate population")


@tool
def get_current_weather(location: str, unit: str = "celsius"):
    """Get the current weather in a given location."""
    return f"Weather in {location} is 25 degrees {unit}."


async def main():
    # Use remote host if set, otherwise defaults in adapter will be used
    remote_host = os.getenv("REMOTE_HOST", "localhost")
    target_url = f"http://{remote_host}:8060"

    print(f"🚀 Starting Gateway Adapter Test (Target: {target_url})...\n")

    llm = GatewayChatModel(base_url=target_url)

    # 1. Health Check
    print("--- 1. Health Check ---")
    is_healthy = await llm.check_health()
    if is_healthy:
        print("✅ Health Check Passed")
    else:
        print("❌ Health Check Failed")
        return

    # 2. Basic Chat
    print("\n--- 2. Basic Chat ---")
    try:
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Hello! Just say 'I am working'."),
        ]
        result = await llm.ainvoke(messages)
        print(f"✅ Response: {result.content}")
    except Exception as e:
        print(f"❌ Basic Chat Failed: {e}")

    # 3. Structured Output
    print("\n--- 3. Structured Output (JSON Mode) ---")
    try:
        structured_llm = llm.with_structured_output(CountryInfo)
        prompt = [HumanMessage(content="Tell me about South Korea.")]

        info: CountryInfo = await structured_llm.ainvoke(prompt)
        print(f"✅ Parsed Pydantic Model: {info}")
        print(f"   Name: {info.name}, Capital: {info.capital}")
    except Exception as e:
        print(f"❌ Structured Output Failed: {e}")

    # 4. Tool Calling
    print("\n--- 4. Tool Calling ---")
    try:
        # Bind tools to the model
        llm_with_tools = llm.bind_tools([get_current_weather])

        tool_prompt = [HumanMessage(content="What is the weather in Seoul?")]

        # First call
        ai_msg = await llm_with_tools.ainvoke(tool_prompt)

        if ai_msg.tool_calls:
            print(f"✅ Tool Calls Received: {len(ai_msg.tool_calls)}")
            for tc in ai_msg.tool_calls:
                print(f"   - Tool: {tc['name']}, Args: {tc['args']}")

            if ai_msg.tool_calls[0]["name"] == "get_current_weather":
                print("   -> Correct tool selected!")
            else:
                print(f"   -> Unexpected tool: {ai_msg.tool_calls[0]['name']}")
        else:
            print(f"⚠️ No tool calls received. Response: {ai_msg.content}")

    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"❌ Tool Calling Failed: {repr(e)}")


if __name__ == "__main__":
    asyncio.run(main())
