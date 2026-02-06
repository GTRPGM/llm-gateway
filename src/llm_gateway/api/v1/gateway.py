from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter()


class GatewayConfig(BaseModel):
    default_provider: str
    override_provider: str | None = None


@router.get("/config", response_model=GatewayConfig)
async def get_config(request: Request):
    router_instance = request.app.state.router
    return GatewayConfig(
        default_provider=router_instance.default_provider,
        override_provider=router_instance.override_provider,
    )


@router.post("/config", response_model=GatewayConfig)
async def update_config(request: Request, config: GatewayConfig):
    router_instance = request.app.state.router
    try:
        router_instance.set_default_provider(config.default_provider)
        router_instance.set_override_provider(config.override_provider)
        return GatewayConfig(
            default_provider=router_instance.default_provider,
            override_provider=router_instance.override_provider,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
