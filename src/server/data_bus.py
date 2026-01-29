from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional

from src.ai.llm_agent import LLMDecision
from src.indicators.indicators import IndicatorSnapshot


@dataclass
class SymbolSnapshot:
    """Represents the latest state of a symbol."""

    symbol: str
    name: str = ""
    ts: datetime = field(default_factory=datetime.now)
    close: float = 0.0
    indicators: Optional[IndicatorSnapshot] = None

    # Rolling T state
    position_state: str = "HOLDING_STOCK"
    t_share: float = 0.0

    # Latest LLM decision
    latest_decision: Optional[LLMDecision] = None

    # Ring buffer for recent close prices for charting
    close_series: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=600))

    # Ring buffer for recent decisions for detail view
    decision_history: Deque[LLMDecision] = field(default_factory=lambda: deque(maxlen=50))


class DataBus:
    """A thread-safe, in-memory store for real-time monitoring data."""

    def __init__(self):
        # Use RLock to allow nested acquisition safely.
        self._lock = threading.RLock()
        self._snapshots: Dict[str, SymbolSnapshot] = {}

    def _get_or_create_snapshot_unlocked(self, symbol: str) -> SymbolSnapshot:
        snap = self._snapshots.get(symbol)
        if snap is None:
            snap = SymbolSnapshot(symbol=symbol)
            self._snapshots[symbol] = snap
        return snap

    def get_or_create_snapshot(self, symbol: str) -> SymbolSnapshot:
        with self._lock:
            return self._get_or_create_snapshot_unlocked(symbol)

    def update_snapshot(self, symbol: str, **kwargs: Any) -> None:
        with self._lock:
            snapshot = self._get_or_create_snapshot_unlocked(symbol)
            for key, value in kwargs.items():
                if hasattr(snapshot, key):
                    setattr(snapshot, key, value)

    def add_close_point(self, symbol: str, ts: datetime, close: float) -> None:
        with self._lock:
            snapshot = self._get_or_create_snapshot_unlocked(symbol)
            snapshot.close_series.append({"ts": ts.isoformat(), "close": close})

    def add_decision(self, symbol: str, decision: LLMDecision) -> None:
        with self._lock:
            snapshot = self._get_or_create_snapshot_unlocked(symbol)
            snapshot.decision_history.append(decision)
            snapshot.latest_decision = decision

    def get_all_snapshots(self) -> List[SymbolSnapshot]:
        with self._lock:
            return list(self._snapshots.values())

    def get_snapshot(self, symbol: str) -> Optional[SymbolSnapshot]:
        with self._lock:
            return self._snapshots.get(symbol)


# Global instance to be shared between the main loop and the server
data_bus = DataBus()
