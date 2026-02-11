from __future__ import annotations

import asyncio
import json
import socket
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.server import shared_state
from src.server.shared_state import data_bus


class SymbolSummary(BaseModel):
    symbol: str
    name: str
    ts: datetime
    close: float
    open: Optional[float] = None
    position_state: str
    t_share: float
    total_share: float
    latest_action: Optional[str] = None
    latest_confidence: Optional[float] = None


class DecisionPoint(BaseModel):
    ts: str
    action: str
    confidence: float
    reasons: List[str]
    risks: List[str]
    operation_plan: Dict[str, Any]
    next_decision_point: List[str]


class ChartPoint(BaseModel):
    ts: str
    close: float


class SymbolDetail(BaseModel):
    symbol: str
    name: str
    close_series: List[ChartPoint]
    decisions: List[DecisionPoint]


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._lock = threading.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        with self._lock:
            self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        with self._lock:
            connections = list(self.active_connections)

        to_remove: List[WebSocket] = []
        for connection in connections:
            try:
                await connection.send_text(message)
            except Exception:
                to_remove.append(connection)

        if to_remove:
            with self._lock:
                for c in to_remove:
                    if c in self.active_connections:
                        self.active_connections.remove(c)


app = FastAPI(title="Trading Monitor API", version="0.1.0")
manager = ConnectionManager()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def snapshot_to_summary(snapshot: SymbolSnapshot) -> Dict[str, Any]:
    # Open price is derived from today's first intraday bar in bar_store (Beijing time).
    open_price: Optional[float] = None
    try:
        store = shared_state.bar_store
        if store is not None:
            win = store.get_window(snapshot.symbol, 5000)
            if win:
                import datetime as dt

                beijing_now = datetime.now(dt.timezone(dt.timedelta(hours=8)))
                today = beijing_now.date()
                for b in win:
                    # b.ts is naive datetime representing Beijing time
                    if b.ts.date() != today:
                        continue
                    open_price = float(b.open)
                    break
    except Exception:
        open_price = None

    return {
        "symbol": snapshot.symbol,
        "name": snapshot.name,
        "ts": snapshot.ts.isoformat(),
        "close": snapshot.close,
        "open": open_price,
        "position_state": snapshot.position_state,
        "t_share": snapshot.t_share,
        "total_share": 1.0 + snapshot.t_share,
        "latest_action": snapshot.latest_decision.action if snapshot.latest_decision else None,
        "latest_confidence": snapshot.latest_decision.confidence if snapshot.latest_decision else None,
    }


@app.get("/api/symbols", response_model=List[SymbolSummary])
async def list_symbols():
    snapshots = data_bus.get_all_snapshots()
    return [snapshot_to_summary(s) for s in snapshots]


@app.get("/api/symbols/{symbol}")
async def get_symbol_detail(symbol: str):
    snapshot = data_bus.get_snapshot(symbol)
    if not snapshot:
        # Try to restore snapshot if it exists in bar_store but not in memory
        store = shared_state.bar_store
        if store and symbol in store._data:
            snapshot = data_bus.get_or_create_snapshot(symbol)
            # Basic info from store
            snapshot.name = symbol # Fallback
            win = store.get_window(symbol, 600)
            if win:
                snapshot.ts = win[-1].ts
                snapshot.close = win[-1].close
                for b in win:
                    data_bus.add_close_point(symbol, b.ts, b.close)
        else:
            return {"error": f"Symbol {symbol} not found"}

    # Restore signal history if memory is empty
    if not snapshot.decision_history:
        store = shared_state.bar_store
        if store:
            signals = store.get_signals(symbol)
            from src.ai.llm_agent import LLMDecision
            for sig in signals:
                decision = LLMDecision(
                    action=str(sig["llm_action"]),
                    confidence=float(sig["llm_confidence"]),
                    reasons=str(sig["llm_reasons"]).split("; "),
                    risks=[],
                    operation_plan={},
                    next_decision_point=[],
                    position_state="",
                    ts=datetime.fromisoformat(str(sig["ts"]))
                )
                data_bus.add_decision(symbol, decision)

    close_series = [
        {"ts": point["ts"], "close": point["close"]} for point in snapshot.close_series
    ]

    decisions = []
    for decision in snapshot.decision_history:
        decisions.append(
            {
                "ts": decision.ts.isoformat(),
                "action": decision.action,
                "confidence": float(decision.confidence),
                "reasons": decision.reasons,
                "risks": decision.risks,
                "operation_plan": decision.operation_plan,
                "next_decision_point": decision.next_decision_point,
            }
        )

    return {
        "symbol": snapshot.symbol,
        "name": snapshot.name,
        "close_series": close_series,
        "decisions": decisions,
    }


@app.get("/api/series/{symbol}")
async def get_series(symbol: str):
    snapshot = data_bus.get_snapshot(symbol)
    if not snapshot:
        return {"error": f"Symbol {symbol} not found"}

    return [
        {"ts": point["ts"], "close": point["close"]} for point in snapshot.close_series
    ]


@app.get("/api/full_series/{symbol}")
async def get_full_series(symbol: str, limit: int = Query(800, ge=1, le=5000)):
    """Return full intraday 1-min OHLC series for the given symbol.

    Data source: BarStore from the running main process (shared_state.bar_store).
    """
    store = shared_state.bar_store
    if store is None:
        return {"error": "bar_store_not_ready"}

    win = store.get_window(symbol, int(limit))
    if not win:
        return []

    # Filter to today's bars in Beijing time
    import datetime as dt
    beijing_now = datetime.now(dt.timezone(dt.timedelta(hours=8)))
    today = beijing_now.date()
    out = []
    for b in win:
        try:
            # b.ts is naive datetime representing Beijing time
            if b.ts.date() != today:
                continue
            out.append(
                {
                    "ts": b.ts.isoformat(timespec="seconds"),
                    "open": float(b.open),
                    "high": float(b.high),
                    "low": float(b.low),
                    "close": float(b.close),
                    "volume": float(getattr(b, "volume", 0.0) or 0.0),
                }
            )
        except Exception:
            continue

    return out


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        snapshots = data_bus.get_all_snapshots()
        updates = [snapshot_to_summary(s) for s in snapshots]
        await websocket.send_text(json.dumps({"type": "update", "data": updates}))

        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


async def broadcast_updates(interval_seconds: float = 5.0):
    while True:
        try:
            snapshots = data_bus.get_all_snapshots()
            updates = [snapshot_to_summary(s) for s in snapshots]
            await manager.broadcast(json.dumps({"type": "update", "data": updates}))
        except Exception as e:
            print(f"Error broadcasting updates: {e}")

        await asyncio.sleep(max(1.0, float(interval_seconds)))


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(broadcast_updates())


def _port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        return s.connect_ex((host, int(port))) == 0


def start_monitor_server(host: str = "127.0.0.1", port: int = 18080) -> Optional[threading.Thread]:
    if _port_in_use(host, port):
        print(f"Monitor server already running at http://{host}:{port} (skip start)")
        return None

    def run() -> None:
        uvicorn.run(app, host=host, port=port, log_level="info")

    thread = threading.Thread(target=run, daemon=True, name="monitor-server")
    thread.start()
    print(f"Monitor server started at http://{host}:{port}")
    return thread
