import asyncio
import json
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional
import httpx
from fastapi import HTTPException, status, Request
from app.config import Settings

logger = logging.getLogger("aalo.ollama_client")

class OllamaClient:
    def __init__(self, settings: Settings):
        self.base_url = settings.ollama_base_url.rstrip("/")
        # Generous read timeout for cold model loading
        self.timeout = httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=30.0)
        # Registry to hold cancellation events for active tasks
        self.active_tasks: Dict[str, asyncio.Event] = {}

    def abort_inference(self, task_id: Optional[str] = None) -> int:
        """Trigger the abort event for a specific task, or all active tasks."""
        count = 0
        if task_id and task_id in self.active_tasks:
            self.active_tasks[task_id].set()
            count = 1
        elif not task_id:
            # If no specific ID is provided, abort everything
            for event in self.active_tasks.values():
                event.set()
                count += 1
        return count

    async def check_health(self) -> Dict[str, Any]:
        """Check if local Ollama daemon is reachable and measure latency."""
        start_time = time.time()
        try:
            async with httpx.AsyncClient(base_url=self.base_url, timeout=5.0) as client:
                resp = await client.get("/api/version")
                latency_ms = round((time.time() - start_time) * 1000, 2)
                if resp.status_code == 200:
                    data = resp.json()
                    return {
                        "status": "connected",
                        "ollama_version": data.get("version", "unknown"),
                        "latency_ms": latency_ms,
                        "url": self.base_url,
                    }
                else:
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
            return {
                "status": "error",
                "error": str(e),
                "url": self.base_url,
            }

    async def list_models(self) -> List[Dict[str, Any]]:
        """Fetch installed models from Ollama /api/tags."""
        try:
            async with httpx.AsyncClient(base_url=self.base_url, timeout=10.0) as client:
                resp = await client.get("/api/tags")
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
                detail=f"Ollama is unreachable at {self.base_url}. Is Ollama running on your Server?",
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to fetch models from Ollama: {str(e)}",
            )

    async def stream_chat(self, payload: Dict[str, Any], request: Request, task_id: str) -> AsyncGenerator[str, None]:
        payload["stream"] = True
        url = f"{self.base_url}/api/chat"
        
        # Create and register the abort event for this specific stream
        abort_event = asyncio.Event()
        self.active_tasks[task_id] = abort_event

        async def get_next(iterator):
            return await iterator.__anext__()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", url, json=payload) as response:
                    if response.status_code != 200:
                        error_body = await response.aread()
                        yield f"data: {json.dumps({'error': error_body.decode('utf-8', errors='replace')})}\n\n"
                        return

                    disconnect_task = asyncio.create_task(request.is_disconnected())
                    abort_task = asyncio.create_task(abort_event.wait())
                    line_iterator = response.aiter_lines()
                    line_task = None

                    try:
                        while True:
                            if line_task is None or line_task.done():
                                line_task = asyncio.create_task(get_next(line_iterator))

                            # Wait for a new line, client disconnect, OR explicit abort request
                            done, pending = await asyncio.wait(
                                [line_task, disconnect_task, abort_task],
                                return_when=asyncio.FIRST_COMPLETED
                            )
                            
                            if abort_task in done:
                                logger.info(f"Explicit abort requested for task {task_id}. Stopping stream.")
                                break # Breaks loop, closes httpx context, kills Ollama inference

                            if disconnect_task in done:
                                if disconnect_task.result():
                                    logger.info("Client disconnected natively. Stopping stream.")
                                    break 
                                else:
                                    disconnect_task = asyncio.create_task(request.is_disconnected())

                            if line_task in done:
                                try:
                                    line = line_task.result()
                                    if line:
                                        yield f"data: {line}\n\n"
                                except StopAsyncIteration:
                                    break
                    finally:
                        # Clean up pending tasks
                        if line_task and not line_task.done():
                            line_task.cancel()
                        if not disconnect_task.done():
                            disconnect_task.cancel()
                        if not abort_task.done():
                            abort_task.cancel()

        except asyncio.CancelledError:
            logger.info("Streaming request was natively cancelled.")
            raise
        except Exception as e:
            logger.exception("Error in stream_chat")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            # Remove the event from the registry when the stream is entirely done
            self.active_tasks.pop(task_id, None)

    async def chat_non_streaming(self, payload: Dict[str, Any], request: Request, task_id: str) -> Dict[str, Any]:
        """Non-streaming chat completion request with disconnect aborting."""
        payload["stream"] = False
        url = f"{self.base_url}/api/chat"

        # Create and register the abort event for this specific stream
        abort_event = asyncio.Event()
        self.active_tasks[task_id] = abort_event
        
        async def perform_request():
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code != 200:
                    raise HTTPException(status_code=resp.status_code, detail=resp.text)
                return resp.json()

        req_task = asyncio.create_task(perform_request())
        disconnect_task = asyncio.create_task(request.is_disconnected())
        abort_task = asyncio.create_task(abort_event.wait())
        
        try:
            while True:
                done, pending = await asyncio.wait(
                    [req_task, disconnect_task, abort_task], 
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                if abort_task in done:
                    logger.info(f"Explicit abort requested for task {task_id}. Stopping inference.")
                    break
                    
                if disconnect_task in done:
                    if disconnect_task.result():
                        logger.info("Client disconnected. Aborting non-streaming inference.")
                        # 499 is standard HTTP for "Client Closed Request"
                        raise HTTPException(status_code=499, detail="Client Closed Request")
                    else:
                        disconnect_task = asyncio.create_task(request.is_disconnected())
                        
                if req_task in done:
                    return req_task.result()
                    
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
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Ollama chat error: {str(e)}",
            )
        finally:
            if not req_task.done():
                req_task.cancel()
            if not disconnect_task.done():
                disconnect_task.cancel()
