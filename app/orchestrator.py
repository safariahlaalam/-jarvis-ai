"""Multi-Agent Orchestrator & Parallel Workers for Jarvis."""
import asyncio
import json
import os
import re
from typing import Any

import boto3
from botocore.config import Config
from openai import AsyncOpenAI, OpenAIError
import psycopg

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_WORKER_MODEL = "qwen/qwen3.8-27b"


def _clean_text(text: str) -> str:
    """Remove hidden reasoning tags if returned by reasoning models."""
    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    return cleaned.strip()


def _extract_json(text: str, fallback_prompt: str) -> dict[str, Any]:
    cleaned = _clean_text(text)
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict) and isinstance(data.get("tasks"), list):
                return data
        except json.JSONDecodeError:
            pass
    return {
        "tasks": [
            {"agent": "analyst", "instruction": fallback_prompt}
        ]
    }


def _fetch_projects_context() -> str:
    db_url = os.getenv("DATABASE_URL", "").strip()
    if not db_url:
        return "Database is not configured."
    try:
        with psycopg.connect(db_url, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM projects LIMIT 25;")
                rows = cur.fetchall()
                cols = [desc[0] for desc in (cur.description or [])]
                items = [dict(zip(cols, row)) for row in rows]
                return json.dumps(items, default=str, ensure_ascii=False)
    except Exception as exc:
        return f"Could not read projects table: {exc}"


def _fetch_files_context() -> str:
    access_key = os.getenv("R2_ACCESS_KEY_ID", "").strip()
    secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
    bucket_name = os.getenv("R2_BUCKET_NAME", "").strip()
    endpoint_url = os.getenv("R2_ENDPOINT_URL", "").strip()
    if not all([access_key, secret_key, bucket_name, endpoint_url]):
        return "Object storage is not configured."

    region_name = "us-east-1"
    host = endpoint_url.replace("https://", "").replace("http://", "").split("/")[0]
    parts = host.split(".")
    if len(parts) >= 3 and parts[0] == "s3":
        region_name = parts[1]

    try:
        client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=region_name,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
            ),
        )
        resp = client.list_objects_v2(Bucket=bucket_name, MaxKeys=50)
        files = [
            {
                "key": item["Key"],
                "size_bytes": item["Size"],
                "last_modified": item["LastModified"].isoformat(),
            }
            for item in resp.get("Contents", [])
        ]
        return json.dumps(files, ensure_ascii=False)
    except Exception as exc:
        return f"Could not list files: {exc}"


class JarvisOrchestrator:
    def __init__(self) -> None:
        api_key = (
            os.getenv("AI_API_KEY", "").strip()
            or os.getenv("GROQ_API_KEY", "").strip()
        )
        if not api_key:
            raise ValueError("AI_API_KEY is not configured.")

        base_url = os.getenv("AI_BASE_URL", "").strip() or DEFAULT_BASE_URL
        self.worker_model = os.getenv("AI_MODEL", "").strip() or DEFAULT_WORKER_MODEL
        self.orchestrator_model = (
            os.getenv("ORCHESTRATOR_MODEL", "").strip() or self.worker_model
        )
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def _plan(self, prompt: str, history: list[dict[str, str]]) -> list[dict[str, str]]:
        planner_system = (
            "You are the Orchestrator of Jarvis, a multi-agent AI system. "
            "Analyze the user's request and split it into 1 to 3 focused subtasks assigned to specialist workers:\n"
            "- 'projects': reads/analyzes user projects stored in PostgreSQL.\n"
            "- 'files': inspects/organizes user files stored in Backblaze B2 cloud storage.\n"
            "- 'analyst': handles reasoning, coding, research, planning, or general answers.\n\n"
            "Return ONLY valid JSON in this exact schema:\n"
            '{"tasks": [{"agent": "projects|files|analyst", "instruction": "specific task description"}]}'
        )
        messages = [{"role": "system", "content": planner_system}]
        messages.extend(history[-6:])
        messages.append({"role": "user", "content": prompt})

        completion = await self.client.chat.completions.create(
            model=self.orchestrator_model,
            messages=messages,
            temperature=0.1,
        )
        raw = completion.choices[0].message.content if completion.choices else ""
        parsed = _extract_json(raw or "", prompt)
        tasks = parsed.get("tasks", [])[:3]
        return [
            {
                "agent": str(t.get("agent", "analyst")).lower().strip(),
                "instruction": str(t.get("instruction", prompt)).strip(),
            }
            for t in tasks
            if t.get("instruction")
        ]

    async def _run_worker(self, task: dict[str, str], original_prompt: str) -> dict[str, str]:
        agent = task["agent"]
        instruction = task["instruction"]

        if agent == "projects":
            context_data = await asyncio.to_thread(_fetch_projects_context)
            system_prompt = (
                "You are the Projects Specialist Agent for Jarvis. "
                "Use the live PostgreSQL projects data below to answer accurately.\n"
                f"LIVE_PROJECTS_JSON:\n{context_data}"
            )
        elif agent == "files":
            context_data = await asyncio.to_thread(_fetch_files_context)
            system_prompt = (
                "You are the Cloud Storage Specialist Agent for Jarvis. "
                "Use the live Backblaze B2 file listing below to answer accurately.\n"
                f"LIVE_FILES_JSON:\n{context_data}"
            )
        else:
            agent = "analyst"
            system_prompt = (
                "You are the Senior Analyst & Engineering Agent for Jarvis. "
                "Provide a thorough, accurate, and structured solution to the assigned subtask."
            )

        completion = await self.client.chat.completions.create(
            model=self.worker_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Original user request: {original_prompt}\nYour specific subtask: {instruction}",
                },
            ],
            temperature=0.3,
        )
        raw = completion.choices[0].message.content if completion.choices else ""
        return {
            "agent": agent,
            "instruction": instruction,
            "output": _clean_text(raw or ""),
        }

    async def _synthesize(
        self,
        prompt: str,
        history: list[dict[str, str]],
        worker_results: list[dict[str, str]],
    ) -> str:
        if len(worker_results) == 1:
            return worker_results[0]["output"]

        compiled = "\n\n".join(
            f"[{res['agent'].upper()} AGENT]\nSubtask: {res['instruction']}\nResult:\n{res['output']}"
            for res in worker_results
        )
        synth_system = (
            "You are the Jarvis Orchestrator. Synthesize the specialist agent outputs below "
            "into one cohesive, clear, well-structured final response in the same language as the user's request. "
            "Do not mention internal agent tags unless helpful."
        )
        messages = [{"role": "system", "content": synth_system}]
        messages.extend(history[-4:])
        messages.append(
            {
                "role": "user",
                "content": f"User Request:\n{prompt}\n\nSpecialist Outputs:\n{compiled}",
            }
        )

        completion = await self.client.chat.completions.create(
            model=self.orchestrator_model,
            messages=messages,
            temperature=0.3,
        )
        raw = completion.choices[0].message.content if completion.choices else ""
        return _clean_text(raw or "")

    async def run(self, prompt: str, history: list[dict[str, str]]) -> dict[str, Any]:
        tasks = await self._plan(prompt, history)
        worker_results = await asyncio.gather(
            *(self._run_worker(task, prompt) for task in tasks)
        )
        final_reply = await self._synthesize(prompt, history, list(worker_results))
        return {
            "reply": final_reply,
            "orchestrator_model": self.orchestrator_model,
            "worker_model": self.worker_model,
            "agents_used": [w["agent"] for w in worker_results],
            "steps": worker_results,
        }
