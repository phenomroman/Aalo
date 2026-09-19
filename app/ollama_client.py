import asyncio
import json
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional
import httpx
from fastapi import HTTPException, Request, status
from app.config import Settings

logger = logging.getLogger("aalo.ollama_client")


class OllamaClient:
    def __init__(self, settings: Settings):
        self.base_url = settings.ollama_base_url.rstrip("/")
        # Generous read timeout for cold model loading
        self.timeout = httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=30.0)
        self._client: Optional[httpx.AsyncClient] = None
        # Registry to hold cancellation events for active tasks
        self.active_tasks: Dict[str, asyncio.Event] = {}

    @property
    def client(self) -> httpx.AsyncClient:
        """Return the active HTTP client or lazily instantiate one with connection pooling."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
            )
        return self._client

    async def start(self) -> None:
        """Initialize the client connection pool."""
        _ = self.client

    async def close(self) -> None:
        """Close the underlying client connection pool cleanly."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def abort_inference(self, task_id: Optional[str] = None) -> int:
        """Trigger the abort event for a specific task, or all active tasks."""
        count = 0
        if task_id and task_id in self.active_tasks:
            self.active_tasks[task_id].set()
            count = 1
        elif not task_id:
            for event in self.active_tasks.values():
                event.set()
                count += 1
        return count

    async def check_health(self) -> Dict[str, Any]:
        """Check if local Ollama daemon is reachable and measure latency."""
        start_time = time.time()
        try:
            resp = await self.client.get("/api/version", timeout=5.0)
            latency_ms = round((time.time() - start_time) * 1000, 2)
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "status": "connected",
                    "ollama_version": data.get("version", "unknown"),
                    "latency_ms": latency_ms,
                    "url": self.base_url,
                }
            return {
                "status": "error",
                "status_code": resp.status_code,
                "latency_ms": latency_ms,
                "url": self.base_url,
            }
        except httpx.ConnectError:
            return {
                "status": "disconnected",
                "error": f"Cannot connect to Ollama at {self.base_url}. Ensure 'ollama serve' is running.",
                "url": self.base_url,
            }
        except Exception as e:
            logger.warning("Error checking Ollama health: %s", e)
            return {
                "status": "error",
                "error": "Ollama service health check failed.",
                "url": self.base_url,
            }

    async def list_models(self) -> List[Dict[str, Any]]:
        """Fetch installed models from Ollama /api/tags."""
        try:
            resp = await self.client.get("/api/tags", timeout=10.0)
            if resp.status_code != 200:
                raise HTTPException(
                    status_code=resp.status_code,
                    detail=f"Ollama returned error {resp.status_code}: {resp.text}",
                )
            data = resp.json()
            return data.get("models", [])
        except httpx.ConnectError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Ollama is unreachable at {self.base_url}. Ensure Ollama is running.",
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Failed to fetch models from Ollama")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to fetch models from local Ollama.",
            )

    async def stream_chat(
        self, payload: Dict[str, Any], request: Request, task_id: str
    ) -> AsyncGenerator[str, None]:
        payload["stream"] = True
        abort_event = asyncio.Event()
        self.active_tasks[task_id] = abort_event

        try:
            req = self.client.build_request("POST", "/api/chat", json=payload)
            response = await self.client.send(req, stream=True)

            if response.status_code != 200:
                error_body = await response.aread()
                yield f"data: {json.dumps({'error': error_body.decode('utf-8', errors='replace')})}\n\n"
                await response.aclose()
                return

            line_iterator = response.aiter_lines()
            try:
                while not abort_event.is_set():
                    read_task = asyncio.create_task(line_iterator.__anext__())
                    abort_task = asyncio.create_task(abort_event.wait())

                    done, pending = await asyncio.wait(
                        [read_task, abort_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for p in pending:
                        p.cancel()

                    if abort_task in done:
                        logger.info("Explicit abort requested for task %s. Stopping stream.", task_id)
                        break

                    if read_task in done:
                        try:
                            line = read_task.result()
                            if line:
                                yield f"data: {line}\n\n"
                        except StopAsyncIteration:
                            break
            finally:
                await response.aclose()

        except asyncio.CancelledError:
            logger.info("Streaming request was cancelled for task %s.", task_id)
            raise
        except Exception as e:
            logger.exception("Error in stream_chat for task %s", task_id)
            yield f"data: {json.dumps({'error': 'An internal error occurred during generation.'})}\n\n"
        finally:
            self.active_tasks.pop(task_id, None)

    async def chat_non_streaming(
        self, payload: Dict[str, Any], request: Request, task_id: str
    ) -> Dict[str, Any]:
        """Non-streaming chat completion request with disconnect and abort handling."""
        payload["stream"] = False
        abort_event = asyncio.Event()
        self.active_tasks[task_id] = abort_event

        req_task = asyncio.create_task(self.client.post("/api/chat", json=payload))
        abort_task = asyncio.create_task(abort_event.wait())
        disconnect_task = asyncio.create_task(request.is_disconnected())

        try:
            done, pending = await asyncio.wait(
                [req_task, abort_task, disconnect_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for p in pending:
                p.cancel()

            if abort_task in done:
                logger.info("Explicit abort requested for task %s.", task_id)
                raise HTTPException(status_code=499, detail="Inference aborted")

            if disconnect_task in done and disconnect_task.result():
                logger.info("Client disconnected. Aborting non-streaming inference for task %s.", task_id)
                raise HTTPException(status_code=499, detail="Client Closed Request")

            resp = req_task.result()
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            return resp.json()

        except httpx.ConnectError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Ollama is unreachable at {self.base_url}.",
            )
        except HTTPException:
            raise
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("Ollama non-streaming chat error")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Ollama chat error occurred.",
            )
        finally:
            self.active_tasks.pop(task_id, None)

