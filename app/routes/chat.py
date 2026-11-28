"""Chat endpoint."""
from fastapi import APIRouter, Depends, Request
from app.errors import not_configured_response
from app.schemas import ChatRequest
from app.security import declare_openapi_auth

router = APIRouter(dependencies=[Depends(declare_openapi_auth)])


@router.post("/api/chat")
async def create_chat_completion(payload: ChatRequest, request: Request):
    return not_configured_response(
        feature="chat",
        message="Chat provider is not configured.",
        request_id=request.state.request_id,
    )
