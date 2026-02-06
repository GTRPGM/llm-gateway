from typing import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from llm_gateway.schemas.chat import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/completions", response_model=ChatResponse)
async def chat_completions(request: Request, body: ChatRequest):
    try:
        engine = request.app.state.engine
        response = await engine.chat(body)

        if isinstance(response, AsyncIterator):

            async def event_generator():
                async for chunk in response:
                    # Yield SSE format
                    data = chunk.model_dump_json()
                    yield f"data: {data}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        return response

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    # Let other exceptions (including HTTPException from providers) propagate
    # to the global exception handler in main.py
