# chat.py
from fastapi import APIRouter, HTTPException, Request

from llm_gateway.schemas.chat import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/completions", response_model=ChatResponse)
async def chat_completions(request: Request, body: ChatRequest):
    try:
        engine = request.app.state.engine
        return await engine.chat(body)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error") from e
