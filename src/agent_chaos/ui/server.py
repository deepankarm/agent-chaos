"""FastAPI server for agent-chaos dashboard."""

import asyncio
import contextlib
import json
import signal
import threading
from pathlib import Path

import uvicorn
import uvicorn.server
from fastapi import FastAPI, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from agent_chaos.ui.events import event_bus

STATIC_DIR = Path(__file__).parent / "static"

_runs_dir: Path = Path(".agent_chaos_runs")

app = FastAPI(title="🃏 agent-chaos", description="Fault Injection Dashboard")


class NoCacheMiddleware(BaseHTTPMiddleware):
    """Disable caching for static files during development."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static") or request.url.path == "/":
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return response


app.add_middleware(NoCacheMiddleware)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def dashboard():
    """Serve the dashboard."""
    return FileResponse(STATIC_DIR / "index.v4.html")


@app.get("/api/traces")
async def get_traces(include_artifacts: bool = Query(True)):
    """Get all traces (live + artifact runs)."""
    traces = [
        {
            "trace_id": t.trace_id,
            "name": t.name,
            "description": t.description,
            "start_time": t.start_time,
            "end_time": t.end_time,
            "status": t.status,
            "total_calls": t.total_calls,
            "failed_calls": t.failed_calls,
            "fault_count": t.fault_count,
            "spans": [
                {
                    "span_id": s.span_id,
                    "provider": s.provider,
                    "status": s.status,
                    "latency_ms": s.latency_ms,
                    "error": s.error,
                    "events": [json.loads(e.to_json()) for e in s.events],
                }
                for s in t.spans
            ],
            "source": "live",
        }
        for t in event_bus.get_traces()
    ]
    if include_artifacts:
        traces.extend(_load_artifact_traces(_runs_dir))
    return traces


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Push real-time events to connected clients."""
    await websocket.accept()
    queue = await event_bus.subscribe()

    try:
        while True:
            queue_task = asyncio.create_task(queue.get())
            recv_task = asyncio.create_task(websocket.receive())
            done, pending = await asyncio.wait(
                {queue_task, recv_task}, return_when=asyncio.FIRST_COMPLETED
            )
            for t in pending:
                t.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await t

            if recv_task in done:
                if recv_task.result().get("type") == "websocket.disconnect":
                    break
                continue

            await websocket.send_text(queue_task.result().to_json())
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    finally:
        event_bus.unsubscribe(queue)
        with contextlib.suppress(Exception):
            await websocket.close()


def run_server(runs_dir: Path, host: str = "127.0.0.1", port: int = 8765):
    """Run the dashboard server with clean shutdown on Ctrl+C.

    Args:
        runs_dir: Directory containing run artifacts from 'agent-chaos run'.
        host: Host to bind to.
        port: Port to bind to.
    """
    global _runs_dir
    _runs_dir = runs_dir

    print(f"🃏 agent-chaos dashboard: http://{host}:{port}")
    print(f"   runs: {runs_dir}")

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
        lifespan="off",
        timeout_graceful_shutdown=2,
    )
    server = uvicorn.Server(config)
    _printed = False

    def _handle_exit(sig: int, frame):
        nonlocal _printed
        if not _printed:
            _printed = True
            print("🛑 Dashboard stopped.")
        server.handle_exit(sig, frame)

    async def _serve():
        # Install signal handlers that don't re-raise (unlike uvicorn's default).
        if threading.current_thread() is threading.main_thread():
            original = {
                sig: signal.signal(sig, _handle_exit)
                for sig in uvicorn.server.HANDLED_SIGNALS
            }
        else:
            original = {}
        try:
            await server._serve(sockets=None)  # pyright: ignore[reportPrivateUsage]
        finally:
            for sig, handler in original.items():
                signal.signal(sig, handler)

    try:
        asyncio.run(_serve())
    except KeyboardInterrupt:
        if not _printed:
            print("🛑 Dashboard stopped.")


def _load_artifact_traces(runs_dir: Path) -> list[dict]:
    """Load traces from CLI artifacts directory."""
    if not runs_dir.exists():
        return []

    traces: list[dict] = []
    for run_dir in sorted((p for p in runs_dir.iterdir() if p.is_dir()), reverse=True):
        score_path = run_dir / "scorecard.json"
        if not score_path.exists():
            continue
        try:
            report = json.loads(score_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        trace_id = report.get("trace_id") or run_dir.name
        name = report.get("scenario_name") or report.get("scenario") or run_dir.name
        passed = bool(report.get("passed", False))

        spans_by_id: dict[str, dict] = {}
        trace_start_ts: str | None = None
        trace_end_ts: str | None = None
        fault_count = 0

        events_path = run_dir / "events.jsonl"
        if events_path.exists():
            for line in events_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    ev = json.loads(line)
                except Exception:
                    continue

                ev_type = ev.get("type")
                if ev_type == "trace_start":
                    trace_start_ts = ev.get("timestamp") or trace_start_ts
                elif ev_type == "trace_end":
                    trace_end_ts = ev.get("timestamp") or trace_end_ts
                elif ev_type in ("span_start", "span_end"):
                    span = spans_by_id.setdefault(
                        ev.get("span_id", ""),
                        {
                            "span_id": ev.get("span_id", ""),
                            "provider": ev.get("provider", ""),
                            "status": "running",
                            "latency_ms": None,
                            "error": "",
                            "events": [],
                        },
                    )
                    if ev_type == "span_end":
                        data = ev.get("data") or {}
                        span["status"] = "success" if data.get("success") else "error"
                        span["latency_ms"] = data.get("latency_ms")
                        span["error"] = data.get("error") or ""
                else:
                    span_id = ev.get("span_id", "")
                    if span_id:
                        span = spans_by_id.setdefault(
                            span_id,
                            {
                                "span_id": span_id,
                                "provider": ev.get("provider", ""),
                                "status": "running",
                                "latency_ms": None,
                                "error": "",
                                "events": [],
                            },
                        )
                        span["events"].append(ev)
                    if ev_type == "fault_injected":
                        fault_count += 1

        spans = list(spans_by_id.values())
        score = report.get("scorecard") or {}
        traces.append(
            {
                "trace_id": trace_id,
                "name": name,
                "description": report.get("description")
                or report.get("scenario_description")
                or "",
                "start_time": trace_start_ts or "",
                "end_time": trace_end_ts,
                "status": "success" if passed else "error",
                "total_calls": score.get("llm_calls_total", len(spans)),
                "failed_calls": score.get(
                    "llm_calls_failed",
                    sum(1 for s in spans if s.get("status") == "error"),
                ),
                "fault_count": score.get("faults_injected_total", fault_count),
                "spans": spans,
                "source": "artifact",
                "run_dir": str(run_dir),
                "report": report,
            }
        )

    return traces
