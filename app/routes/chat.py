"""Chat & Multi-Agent Orchestrator endpoints."""
import os

from fastapi import APIRouter, Depends, HTTPException, Request
from openai import AsyncOpenAI, OpenAIError

from app.errors import not_configured_response
from app.orchestrator import JarvisOrchestrator, _clean_text
from app.schemas import ChatRequest
from app.security import declare_openapi_auth

router = APIRouter(dependencies=[Depends(declare_openapi_auth)])

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "qwen/qwen3.8-27b"
DEFAULT_SYSTEM_PROMPT = "You are Jarvis, a concise, helpful personal AI assistant."


def _get_ai_client() -> tuple[AsyncOpenAI | None, str]:
    api_key = (
        os.getenv("AI_API_KEY", "").strip()
        or os.getenv("GROQ_API_KEY", "").strip()
    )
    if not api_key:
        return None, ""

    base_url = os.getenv("AI_BASE_URL", "").strip() or DEFAULT_BASE_URL
    model = os.getenv("AI_MODEL", "").strip() or DEFAULT_MODEL
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    return client, model


@router.post("/api/chat")
async def create_chat_completion(payload: ChatRequest, request: Request):
    client, model = _get_ai_client()
    if client is None:
        return not_configured_response(
            feature="chat",
            message="Chat provider is not configured.",
            request_id=request.state.request_id,
        )

    system_prompt = (
        os.getenv("JARVIS_SYSTEM_PROMPT", "").strip() or DEFAULT_SYSTEM_PROMPT
    )
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(
        {"role": msg.role, "content": msg.content} for msg in payload.history
    )
    messages.append({"role": "user", "content": payload.prompt})

    try:
        completion = await client.chat.completions.create(
            model=model,
            messages=messages,
        )
    except OpenAIError as exc:
        raise HTTPException(status_code=502, detail=f"AI provider error: {exc}") from exc

    choice = completion.choices[0] if completion.choices else None
    reply = _clean_text((choice.message.content if choice and choice.message else "") or "")

    usage = None
    if completion.usage is not None:
        usage = {
            "prompt_tokens": completion.usage.prompt_tokens,
            "completion_tokens": completion.usage.completion_tokens,
            "total_tokens": completion.usage.total_tokens,
        }

    return {
        "reply": reply,
        "model": completion.model or model,
        "usage": usage,
    }


@router.post("/api/agent")
async def run_multi_agent(payload: ChatRequest, request: Request):
    try:
        orchestrator = JarvisOrchestrator()
    except ValueError:
        return not_configured_response(
            feature="agent",
            message="Chat provider is not configured.",
            request_id=request.state.request_id,
        )

    history = [{"role": msg.role, "content": msg.content} for msg in payload.history]
    try:
        return await orchestrator.run(prompt=payload.prompt, history=history)
    except OpenAIError as exc:
        raise HTTPException(status_code=502, detail=f"Agent provider error: {exc}") from exc
