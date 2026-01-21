from fastapi import APIRouter, HTTPException

from llm_gateway.schemas.chat import ChatRequest, ChatResponse
from llm_gateway.services.router import llm_router

router = APIRouter()


@router.post("/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest):
    """
    Generates a model response for the given chat conversation.
    Compatible with OpenAI Chat Completion API.
    """
    try:
        response = await llm_router.route_chat_completion(request)
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error") from e
