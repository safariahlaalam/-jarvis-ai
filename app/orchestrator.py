"""Multi-Agent Orchestrator & Specialist Workers for Jarvis."""
import asyncio
from html.parser import HTMLParser
import json
import os
import re
from typing import Any
import urllib.parse
import urllib.request
import uuid

import boto3
from botocore.config import Config
from openai import AsyncOpenAI
import psycopg

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_WORKER_MODEL = "qwen/qwen3.8-27b"


def _clean_text(text: str) -> str:
    """Remove hidden reasoning tags if returned by reasoning models."""
    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    return cleaned.strip()


def _extract_json(text: str, fallback: dict[str, Any]) -> dict[str, Any]:
    cleaned = _clean_text(text)
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    return fallback


class _DDGParser(HTMLParser):
    """Lightweight HTML parser for DuckDuckGo search results."""

    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._in_title = False
        self._in_snippet = False
        self._current_title: list[str] = []
        self._current_snippet: list[str] = []
        self._current_url = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = dict(attrs)
        cls = attr_map.get("class") or ""
        if tag == "a" and "result__a" in cls:
            self._in_title = True
            self._current_title = []
            self._current_url = attr_map.get("href") or ""
        elif tag in ("a", "div", "span") and "result__snippet" in cls:
            self._in_snippet = True
            self._current_snippet = []

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._current_title.append(data)
        elif self._in_snippet:
            self._current_snippet.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_title:
            self._in_title = False
        elif tag in ("a", "div", "span") and self._in_snippet:
            self._in_snippet = False
            title = "".join(self._current_title).strip()
            snippet = "".join(self._current_snippet).strip()
            if title or snippet:
                self.results.append(
                    {
                        "title": title,
                        "url": self._current_url,
                        "snippet": snippet,
                    }
                )


def _search_web_sync(query: str, max_results: int = 5) -> str:
    """Search the live web using Tavily (if configured) or DuckDuckGo fallback."""
    tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
    if tavily_key:
        try:
            payload = json.dumps(
                {
                    "api_key": tavily_key,
                    "query": query,
                    "search_depth": "basic",
                    "max_results": max_results,
                }
            ).encode("utf-8")
            req = urllib.request.Request(
                "https://api.tavily.com/search",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                body = json.loads(resp.read().decode("utf-8", errors="replace"))
                items = [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "snippet": r.get("content", ""),
                    }
                    for r in body.get("results", [])[:max_results]
                ]
                return json.dumps(items, ensure_ascii=False)
        except Exception:
            pass

    try:
        encoded_q = urllib.parse.urlencode({"q": query}).encode("utf-8")
        req = urllib.request.Request(
            "https://html.duckduckgo.com/html/",
            data=encoded_q,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
                ),
                "Content-Type": "application/x-www-form-urlencoded",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            html_text = resp.read().decode("utf-8", errors="replace")
        parser = _DDGParser()
        parser.feed(html_text)
        if parser.results:
            return json.dumps(parser.results[:max_results], ensure_ascii=False)
        return "No live search results matched the query."
    except Exception as exc:
        return f"Web search error: {exc}"


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


def _create_projects_in_db(projects_list: list[dict[str, Any]]) -> str:
    """Inspect the projects table schema and insert new project rows safely."""
    db_url = os.getenv("DATABASE_URL", "").strip()
    if not db_url:
        return "Database is not configured."

    try:
        with psycopg.connect(db_url, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT column_name, column_default, data_type
                    FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = 'projects';
                    """
                )
                schema_rows = cur.fetchall()
                if not schema_rows:
                    return "Table 'projects' was not found in PostgreSQL."

                col_info = {
                    row[0]: {"default": row[1], "type": row[2]} for row in schema_rows
                }
                created_records: list[dict[str, Any]] = []

                for proj in projects_list:
                    title_val = str(
                        proj.get("name") or proj.get("title") or "Untitled Project"
                    ).strip()
                    desc_val = str(proj.get("description") or "").strip()
                    status_val = str(proj.get("status") or "active").strip()

                    row_data: dict[str, Any] = {}
                    if (
                        "id" in col_info
                        and col_info["id"]["default"] is None
                        and "uuid" in col_info["id"]["type"].lower()
                    ):
                        row_data["id"] = str(uuid.uuid4())

                    if "name" in col_info:
                        row_data["name"] = title_val
                    elif "title" in col_info:
                        row_data["title"] = title_val

                    for desc_col in ("description", "details", "summary", "content"):
                        if desc_col in col_info:
                            row_data[desc_col] = desc_val
                            break

                    if "status" in col_info:
                        row_data["status"] = status_val

                    if not row_data:
                        return f"No compatible columns found in projects table: {list(col_info.keys())}"

                    cols_sql = ", ".join(row_data.keys())
                    placeholders = ", ".join(["%s"] * len(row_data))
                    cur.execute(
                        f"INSERT INTO projects ({cols_sql}) VALUES ({placeholders}) RETURNING *;",
                        list(row_data.values()),
                    )
                    inserted = cur.fetchone()
                    ret_cols = [d[0] for d in (cur.description or [])]
                    if inserted:
                        created_records.append(dict(zip(ret_cols, inserted)))

            conn.commit()
        return json.dumps(created_records, default=str, ensure_ascii=False)
    except Exception as exc:
        return f"Failed to insert project into PostgreSQL: {exc}"


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

    async def _plan(
        self, prompt: str, history: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        planner_system = (
            "You are the Orchestrator of Jarvis, a multi-agent AI system. "
            "Analyze the user's request and split it into 1 to 4 focused subtasks assigned to specialist workers:\n"
            "- 'web_search': searches the live internet for current facts, news, tools, or external data.\n"
            "- 'projects': reads existing projects stored in PostgreSQL.\n"
            "- 'project_creator': creates and saves new project(s) into PostgreSQL when the user asks to create/save/add a project.\n"
            "- 'files': inspects/organizes user files stored in Backblaze B2 cloud storage.\n"
            "- 'analyst': handles reasoning, coding, planning, or general answers.\n\n"
            "Return ONLY valid JSON in this exact schema:\n"
            '{"tasks": [{"agent": "web_search|projects|project_creator|files|analyst", "instruction": "specific task or search query"}]}'
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
        parsed = _extract_json(
            raw or "",
            {"tasks": [{"agent": "analyst", "instruction": prompt}]},
        )
        tasks = parsed.get("tasks", [])[:4]
        valid_agents = {
            "web_search",
            "projects",
            "project_creator",
            "files",
            "analyst",
        }
        cleaned_tasks = []
        for t in tasks:
            if not isinstance(t, dict) or not t.get("instruction"):
                continue
            agent_name = str(t.get("agent", "analyst")).lower().strip()
            if agent_name not in valid_agents:
                agent_name = "analyst"
            cleaned_tasks.append(
                {
                    "agent": agent_name,
                    "instruction": str(t["instruction"]).strip(),
                }
            )
        return cleaned_tasks or [{"agent": "analyst", "instruction": prompt}]

    async def _run_project_creator(
        self,
        instruction: str,
        original_prompt: str,
        prior_context: str,
    ) -> dict[str, str]:
        extractor_system = (
            "You are the Project Creator Agent for Jarvis. "
            "Extract the project(s) to create from the user's request and any prior research context. "
            "Return ONLY valid JSON in this exact format:\n"
            '{"projects": [{"name": "Concise Project Title", "description": "Detailed actionable description", "status": "active"}]}'
        )
        user_msg = (
            f"Original Request: {original_prompt}\n"
            f"Task Instruction: {instruction}\n"
        )
        if prior_context:
            user_msg += f"\nContext from earlier agents:\n{prior_context}"

        completion = await self.client.chat.completions.create(
            model=self.worker_model,
            messages=[
                {"role": "system", "content": extractor_system},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
        )
        raw = completion.choices[0].message.content if completion.choices else ""
        parsed = _extract_json(
            raw or "",
            {
                "projects": [
                    {
                        "name": instruction[:80],
                        "description": original_prompt,
                        "status": "active",
                    }
                ]
            },
        )
        projects_to_create = parsed.get("projects")
        if not isinstance(projects_to_create, list) or not projects_to_create:
            projects_to_create = [
                {
                    "name": instruction[:80],
                    "description": original_prompt,
                    "status": "active",
                }
            ]

        db_result = await asyncio.to_thread(
            _create_projects_in_db, projects_to_create[:5]
        )
        return {
            "agent": "project_creator",
            "instruction": instruction,
            "output": f"Database insert result:\n{db_result}",
        }

    async def _run_worker(
        self,
        task: dict[str, str],
        original_prompt: str,
        prior_context: str = "",
    ) -> dict[str, str]:
        agent = task["agent"]
        instruction = task["instruction"]

        if agent == "project_creator":
            return await self._run_project_creator(
                instruction, original_prompt, prior_context
            )

        if agent == "web_search":
            # Generate a focused search query first so web results are crisp
            q_completion = await self.client.chat.completions.create(
                model=self.orchestrator_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Convert the instruction into a concise, high-signal web search query "
                            "(3 to 8 words, no quotes or commentary)."
                        ),
                    },
                    {"role": "user", "content": instruction},
                ],
                temperature=0.0,
            )
            search_query = _clean_text(
                (q_completion.choices[0].message.content if q_completion.choices else "")
                or instruction
            )
            search_results = await asyncio.to_thread(_search_web_sync, search_query)
            system_prompt = (
                "You are the Live Web Research Specialist Agent for Jarvis. "
                "Synthesize accurate, factual findings from the live search results below and include key source URLs.\n"
                f"SEARCH_QUERY: {search_query}\n"
                f"LIVE_WEB_RESULTS_JSON:\n{search_results}"
            )
        elif agent == "projects":
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
                    "content": (
                        f"Original user request: {original_prompt}\n"
                        f"Your specific subtask: {instruction}"
                    ),
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
        if len(worker_results) == 1 and worker_results[0]["agent"] != "project_creator":
            return worker_results[0]["output"]

        compiled = "\n\n".join(
            f"[{res['agent'].upper()} AGENT]\nSubtask: {res['instruction']}\nResult:\n{res['output']}"
            for res in worker_results
        )
        synth_system = (
            "You are the Jarvis Orchestrator. Synthesize the specialist agent outputs below "
            "into one cohesive, clear, well-structured final response in the same language as the user's request. "
            "If new projects were created in the database, clearly confirm their saved names and details."
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

        # Wave 1: Run read/research/analysis workers in parallel
        wave1_tasks = [t for t in tasks if t["agent"] != "project_creator"]
        wave2_tasks = [t for t in tasks if t["agent"] == "project_creator"]

        wave1_results: list[dict[str, str]] = []
        if wave1_tasks:
            wave1_results = list(
                await asyncio.gather(
                    *(self._run_worker(task, prompt) for task in wave1_tasks)
                )
            )

        # Wave 2: Run project_creator with Wave 1 outputs as context
        wave2_results: list[dict[str, str]] = []
        if wave2_tasks:
            prior_context = "\n\n".join(
                f"[{r['agent'].upper()}]: {r['output']}" for r in wave1_results
            )
            wave2_results = list(
                await asyncio.gather(
                    *(
                        self._run_worker(task, prompt, prior_context)
                        for task in wave2_tasks
                    )
                )
            )

        all_results = wave1_results + wave2_results
        final_reply = await self._synthesize(prompt, history, all_results)
        return {
            "reply": final_reply,
            "orchestrator_model": self.orchestrator_model,
            "worker_model": self.worker_model,
            "agents_used": [w["agent"] for w in all_results],
            "steps": all_results,
        }
