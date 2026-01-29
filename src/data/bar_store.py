from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Deque, Dict, Iterable, List

from src.indicators.types import Bar


@dataclass
class BarWindow:
    maxlen: int
    bars: Deque[Bar]


class BarStore:
    def __init__(self, window_size: int = 600):
        self._window_size = window_size
        self._data: Dict[str, BarWindow] = {}

    @staticmethod
    def _normalize_ts(ts: datetime) -> datetime:
        # Normalize to naive datetime in UTC to avoid mixing offset-aware/naive datetimes.
        if ts.tzinfo is not None and ts.tzinfo.utcoffset(ts) is not None:
            return ts.astimezone(timezone.utc).replace(tzinfo=None)
        return ts

    def upsert_bars(self, symbol: str, new_bars: Iterable[Bar]) -> None:
        win = self._data.get(symbol)
        if win is None:
            win = BarWindow(maxlen=self._window_size, bars=deque(maxlen=self._window_size))
            self._data[symbol] = win

        existing = {self._normalize_ts(b.ts): i for i, b in enumerate(win.bars)}
        for b in new_bars:
            ts = self._normalize_ts(b.ts)
            b = Bar(
                ts=ts,
                open=b.open,
                high=b.high,
                low=b.low,
                close=b.close,
                volume=b.volume,
                amount=b.amount,
            )
            if ts in existing:
                idx = existing[ts]
                tmp = list(win.bars)
                tmp[idx] = b
                win.bars = deque(tmp, maxlen=win.maxlen)
                existing = {self._normalize_ts(bb.ts): i for i, bb in enumerate(win.bars)}
            else:
                win.bars.append(b)
                existing[ts] = len(win.bars) - 1

        # ensure sorted by ts (some sources may return unordered)
        win.bars = deque(sorted(win.bars, key=lambda x: x.ts), maxlen=win.maxlen)

    def get_window(self, symbol: str, n: int) -> List[Bar]:
        win = self._data.get(symbol)
        if not win:
            return []
        if n <= 0:
            return list(win.bars)
        return list(win.bars)[-n:]

    def latest_ts(self, symbol: str) -> datetime | None:
        win = self._data.get(symbol)
        if not win or not win.bars:
            return None
        return win.bars[-1].ts

