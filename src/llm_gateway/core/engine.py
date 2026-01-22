from llm_gateway.core.interfaces import BaseRouter
from llm_gateway.schemas.chat import ChatRequest, ChatResponse


class LLMEngine:
    def __init__(self, router: BaseRouter):
        self.router = router

    async def chat(self, request: ChatRequest) -> ChatResponse:
        return await self.router.route_chat(request)
