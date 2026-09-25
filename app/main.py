from starlette.middleware import body_limit
import logging
import uuid
from textwrap import dedent
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from fastapi import Body, Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict
from app.auth import verify_api_key
from app.config import get_settings
from app.ollama_client import OllamaClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("aalo")

settings = get_settings()
ollama_client = OllamaClient(settings)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: Optional[List[Dict[str, Any]]] = None
    stream: bool = True
    think: bool = False
    task_id: Optional[str] = None


class AbortRequest(BaseModel):
    task_id: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Aalo Gateway...")
    logger.info("Checking connection to local Ollama on %s...", settings.ollama_base_url)
    await ollama_client.start()
    health = await ollama_client.check_health()
    if health.get("status") == "connected":
        logger.info(
            "Connected to local Ollama (version %s, latency %sms)",
            health.get("ollama_version"),
            health.get("latency_ms"),
        )
    else:
        logger.warning(
            "Local Ollama is not yet reachable at %s. Error: %s",
            settings.ollama_base_url,
            health.get("error"),
        )
    yield
    logger.info("Aalo Gateway shutting down.")
    await ollama_client.close()


app = FastAPI(
    title="Aalo Gateway",
    description="Secure Remote Access Gateway for Local Ollama",
    version="0.1.0",
    lifespan=lifespan,
)

# Enable CORS for cross-origin web apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"status": "Aalo Gateway Running"}


@app.get("/health", summary="Health check for Gateway and local Ollama")
async def health_check():
    """Public health check endpoint. Reports status of both gateway and local Ollama service."""
    ollama_health = await ollama_client.check_health()
    return {
        "status": "online",
        "gateway": "Aalo Gateway v1.0",
        "ollama": ollama_health,
    }


@app.get("/api/models", summary="List locally installed Ollama models")
async def list_models(_auth: str = Depends(verify_api_key)):
    """Fetch all models currently pulled and available."""
    models = await ollama_client.list_models()
    return {"models": models}

# Chat endpoints
@app.post("/api/chat", summary="Chat completion with streaming support")
async def chat_endpoint(
    request: Request,
    body: ChatRequest,
    _auth: str = Depends(verify_api_key),
):
    """Proxy chat requests to local Ollama with real-time SSE streaming or JSON output."""
    task_id = body.task_id or str(uuid.uuid4())
    payload = body.model_dump()
    payload["task_id"] = task_id

    # Add system prompt to payload if not already present
    if not payload.get("messages"):
        payload["messages"] = []
    
    system_prompt = {
    "role": "system",
    "content": dedent("""\
        You are Alo, a highly capable and intelligent AI assistant powering the Aalo application. You were created by Masrur Ahmed Roman at phenomroman.com. 

        Your personality is sharp, pragmatic, and genuinely helpful. You combine executive-level professionalism with a touch of approachable, grounded wit. You speak with the confidence of a seasoned technology and operations partner. You absolutely never use robotic, sycophantic filler phrases (e.g., "As an AI...", "I'd be happy to help!", "Here is the analysis you requested"). You simply deliver the work.

        Core Execution Guidelines:
        1. Text Summarization: Deliver bottom-line-up-front (BLUF) summaries. Strip away fluff and highlight only actionable insights.
        2. Report Writing: Structure documents logically with clear headings, precise bullet points, and executive summaries. Tailor your tone for professional enterprise environments, ensuring documents like Business Requirement Documents (BRDs), vendor assessments, and strategy memos are boardroom-ready.
        3. Data Analysis: Do not just regurgitate numbers—interpret them. Actively look for trends, highlight potential calculation errors, flag data anomalies, and structure your findings to immediately drive decision-making. 
        4. Code Synthesis: Write clean, modular, and thoroughly commented code. Prioritize efficiency, error-handling, and modern best practices, particularly when dealing with Python, data processing pipelines, or web deployments.
        
        Always adapt your response length to the complexity of the prompt: be exceptionally concise for simple queries, and highly structured and comprehensive for deep analytical tasks.\
    """)
    }
    # Strip any existing client-supplied system messages before prepending the master prompt
    payload["messages"] = [
        msg for msg in payload.get("messages", []) 
        if msg.get("role") != "system"
    ]
    payload["messages"].insert(0, system_prompt)

    if body.stream:
        return StreamingResponse(
            ollama_client.stream_chat(payload, request, task_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        result = await ollama_client.chat_non_streaming(payload, request, task_id)
        return JSONResponse(content=result)


@app.post("/api/chat/abort", summary="Manually abort inference")
async def abort_inference_endpoint(
    body: Optional[AbortRequest] = Body(None),
    _auth: str = Depends(verify_api_key),
):
    """Explicitly stop ongoing Ollama inference via frontend Stop button."""
    task_id = body.task_id if body else None
    aborted_count = ollama_client.abort_inference(task_id)

    return {
        "status": "success",
        "detail": "Inference aborted",
        "tasks_aborted": aborted_count,
    }