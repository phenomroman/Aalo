import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict
from fastapi import Depends, FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from app.auth import verify_api_key
from app.config import Settings, get_settings
from app.ollama_client import OllamaClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("aalo")

settings = get_settings()
ollama_client = OllamaClient(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Aalo Gateway...")
    logger.info("Checking connection to local Ollama on %s...", settings.ollama_base_url)
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
    allow_credentials=True,
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
    _auth: str = Depends(verify_api_key),
):
    """Proxy chat requests to local Ollama with real-time SSE streaming or JSON output."""
    try:
        body: Dict[str, Any] = await request.json()
    except Exception:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": "Invalid JSON body"},
        )
    
    # Extract task_id if frontend provides one, otherwise generate one
    task_id = body.get("task_id", str(uuid.uuid4()))

    model = body.get("model")
    if not model:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": "Missing required field 'model'"},
        )

    if "think" not in body:
        body["think"] = False
    else:
        body["think"] = bool(body["think"])

    stream = body.get("stream", True)
    if stream:
        return StreamingResponse(
            ollama_client.stream_chat(body, request, task_id), # <--- PASSED request and task_id HERE
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  
            },
        )
    else:
        result = await ollama_client.chat_non_streaming(body, request, task_id) # <--- PASSED request and task_id HERE
        return JSONResponse(content=result)

@app.post("/api/chat/abort", summary="Manually abort inference")
async def abort_inference_endpoint(
    request: Request,
    _auth: str = Depends(verify_api_key)
):
    """Explicitly stop ongoing Ollama inference via frontend Stop button."""
    try:
        body = await request.json()
        task_id = body.get("task_id")
    except Exception:
        # If no JSON body is sent, default to None
        task_id = None
        
    # If task_id is None, abort ALL active local generations
    aborted_count = ollama_client.abort_inference(task_id)
    
    return {
        "status": "success", 
        "detail": "Inference aborted", 
        "tasks_aborted": aborted_count
    }