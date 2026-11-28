"""Chat, Multi-Agent Orchestrator, and Neural TTS endpoints."""
import os
import re

import edge_tts
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from openai import AsyncOpenAI, OpenAIError
from pydantic import BaseModel, Field

from app.errors import not_configured_response
from app.orchestrator import JarvisOrchestrator, _clean_text
from app.schemas import ChatRequest
from app.security import declare_openapi_auth
from app.services.file_builder import extract_and_build_file


router = APIRouter(dependencies=[Depends(declare_openapi_auth)])

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_CODER_MODEL = "qwen/qwen3.8-27b"
DEFAULT_STRATEGIST_MODEL = "openai/gpt-oss-120b"
DEFAULT_SYSTEM_PROMPT = "You are Jarvis, a concise, helpful personal AI assistant."

FILE_GENERATION_INSTRUCTION = (
    "When the user asks for a file, report, document, plan, or code, wrap the complete file inside "
    "[FILE:exact_filename.ext]...[/FILE] tags and write a brief summary outside the tags. "
    "For .pdf files: start with [STYLE:dark] if the user asks for dark/cyber/RAMI style or [STYLE:executive] for clean white business style, "
    "followed by '# Document Title', '## Section Headings', '> Key Highlight Callouts', '- Bullet points', and '| Markdown | Tables |' "
    "tailored to the exact layout and sections the user requested. "
    "For .html files: generate a complete, modern, responsive single-file HTML5/CSS3 design with clean typography, cards, and color styling."
)

_FILE_INTENT_RE = re.compile(
    r"\b(file|pdf|html|csv|json|script|code|document|report|plan|proposal|export|download|send me|make me|create a)\b",
    re.IGNORECASE,
)


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    voice: str | None = Field(default=None)
    rate: str = Field(default="+0%")
    pitch: str = Field(default="+0Hz")


def _prepare_speech_text(text: str) -> str:
    cleaned = _clean_text(text)
    cleaned = re.sub(r"```[\s\S]*?```", " ", cleaned)
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
    cleaned = re.sub(r"[#*_`~>|]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _pick_default_voice(text: str) -> str:
    if re.search(r"[\u0600-\u06FF]", text):
        return os.getenv("TTS_ARABIC_VOICE", "ar-IQ-RanaNeural").strip()
    return os.getenv("TTS_ENGLISH_VOICE", "en-US-AvaNeural").strip()


def _get_fast_chat_client() -> tuple[AsyncOpenAI | None, str]:
    coder_key = os.getenv("RAMI_CODER_API_KEY", "").strip()
    if coder_key:
        base_url = os.getenv("RAMI_CODER_BASE_URL", "").strip() or GROQ_BASE_URL
        model = os.getenv("RAMI_CODER_MODEL", "").strip() or DEFAULT_CODER_MODEL
        return AsyncOpenAI(api_key=coder_key, base_url=base_url), model

    strategist_key = os.getenv("RAMI_STRATEGIST_API_KEY", "").strip()
    if strategist_key:
        base_url = os.getenv("RAMI_STRATEGIST_BASE_URL", "").strip() or GROQ_BASE_URL
        model = os.getenv("RAMI_STRATEGIST_MODEL", "").strip() or DEFAULT_STRATEGIST_MODEL
        return AsyncOpenAI(api_key=strategist_key, base_url=base_url), model

    return None, ""


@router.post("/api/chat")
async def create_chat_completion(payload: ChatRequest, request: Request):
    client, model = _get_fast_chat_client()
    if client is None:
        return not_configured_response(
            feature="chat",
            message="Fast Chat provider is not configured. Add RAMI_CODER_API_KEY or RAMI_STRATEGIST_API_KEY.",
            request_id=request.state.request_id,
        )

    base_prompt = (
        os.getenv("JARVIS_SYSTEM_PROMPT", "").strip() or DEFAULT_SYSTEM_PROMPT
    )
    system_prompt = f"{base_prompt}\n\n{FILE_GENERATION_INSTRUCTION}"

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
    raw_reply = _clean_text((choice.message.content if choice and choice.message else "") or "")
    reply, file_payload = extract_and_build_file(raw_reply)

    usage = None
    if completion.usage is not None:
        usage = {
            "prompt_tokens": completion.usage.prompt_tokens,
            "completion_tokens": completion.usage.completion_tokens,
            "total_tokens": completion.usage.total_tokens,
        }

    return {
        "reply": reply,
        "file": file_payload,
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
            message="Multi-Agent providers are not configured.",
            request_id=request.state.request_id,
        )

    history = [{"role": msg.role, "content": msg.content} for msg in payload.history]
    agent_prompt = payload.prompt
    if _FILE_INTENT_RE.search(payload.prompt):
        agent_prompt = f"{payload.prompt}\n\n[Formatting rule: {FILE_GENERATION_INSTRUCTION}]"

    try:
        result = await orchestrator.run(prompt=agent_prompt, history=history)
    except OpenAIError as exc:
        raise HTTPException(status_code=502, detail=f"Agent provider error: {exc}") from exc

    if isinstance(result, dict):
        for key in ("reply", "response", "message"):
            if isinstance(result.get(key), str) and result[key]:
                cleaned_text, file_payload = extract_and_build_file(result[key])
                result[key] = cleaned_text
                if file_payload is not None:
                    result["file"] = file_payload
                break

    return result


@router.post("/api/tts")
async def synthesize_speech(payload: TTSRequest):
    speech_text = _prepare_speech_text(payload.text)
    if not speech_text:
        raise HTTPException(status_code=400, detail="No speakable text provided.")

    selected_voice = (payload.voice or "").strip() or _pick_default_voice(speech_text)
    try:
        communicate = edge_tts.Communicate(
            text=speech_text,
            voice=selected_voice,
            rate=payload.rate,
            pitch=payload.pitch,
        )
        audio_bytes = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_bytes.extend(chunk["data"])
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"TTS synthesis failed: {exc}") from exc

    if not audio_bytes:
        raise HTTPException(status_code=502, detail="TTS returned empty audio.")

    return Response(content=bytes(audio_bytes), media_type="audio/mpeg")
