from __future__ import annotations

import sqlite3
import os
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Deque, Dict, Iterable, List, Optional

from src.indicators.types import Bar


@dataclass
class BarWindow:
    maxlen: int
    bars: Deque[Bar]


class BarStore:
    def __init__(self, window_size: int = 600, db_path: Optional[str] = "data/bars.db"):
        self._window_size = window_size
        self._data: Dict[str, BarWindow] = {}
        self._db_path = db_path
        
        if self._db_path:
            os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
            self._init_db()
            self._load_from_db()

    def _init_db(self):
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bars (
                    symbol TEXT,
                    ts TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    amount REAL,
                    PRIMARY KEY (symbol, ts)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_ts ON bars (symbol, ts)")
            
            # 新增信号存储表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    symbol TEXT,
                    ts TEXT,
                    signal_type TEXT,
                    price REAL,
                    rule_reason TEXT,
                    llm_action TEXT,
                    llm_confidence REAL,
                    llm_reasons TEXT,
                    t_share REAL,
                    PRIMARY KEY (symbol, ts, signal_type)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sig_symbol_ts ON signals (symbol, ts)")

    def save_signal(self, symbol: str, ts: datetime, signal_type: str, price: float, 
                    rule_reason: str = "", llm_action: str = "", llm_confidence: float = 0.0, 
                    llm_reasons: str = "", t_share: float = 0.0):
        """保存买卖点信号到数据库"""
        if not self._db_path:
            return
            
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                INSERT INTO signals (symbol, ts, signal_type, price, rule_reason, llm_action, llm_confidence, llm_reasons, t_share)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, ts, signal_type) DO UPDATE SET
                    price=excluded.price,
                    rule_reason=excluded.rule_reason,
                    llm_action=excluded.llm_action,
                    llm_confidence=excluded.llm_confidence,
                    llm_reasons=excluded.llm_reasons,
                    t_share=excluded.t_share
            """, (symbol, ts.isoformat(), signal_type, float(price), rule_reason, llm_action, llm_confidence, llm_reasons, t_share))

    def _load_from_db(self):
        """Load recent bars from SQLite into memory windows."""
        with sqlite3.connect(self._db_path) as conn:
            # For each symbol, get the last window_size bars
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT symbol FROM bars")
            symbols = [row[0] for row in cursor.fetchall()]
            
            for symbol in symbols:
                cursor.execute("""
                    SELECT ts, open, high, low, close, volume, amount 
                    FROM bars 
                    WHERE symbol = ? 
                    ORDER BY ts DESC 
                    LIMIT ?
                """, (symbol, self._window_size))
                
                rows = cursor.fetchall()
                bars = []
                for row in reversed(rows):
                    bars.append(Bar(
                        ts=datetime.fromisoformat(row[0]),
                        open=row[1],
                        high=row[2],
                        low=row[3],
                        close=row[4],
                        volume=row[5],
                        amount=row[6]
                    ))
                
                if bars:
                    self._data[symbol] = BarWindow(
                        maxlen=self._window_size, 
                        bars=deque(bars, maxlen=self._window_size)
                    )

    @staticmethod
    def _normalize_ts(ts: datetime) -> datetime:
        # A-share data is Beijing time. We keep it as naive datetime representing UTC+8.
        if ts.tzinfo is not None:
            # If it has timezone, convert to UTC+8 then make naive
            import datetime as dt
            beijing_tz = dt.timezone(dt.timedelta(hours=8))
            return ts.astimezone(beijing_tz).replace(tzinfo=None)
        return ts

    def upsert_bars(self, symbol: str, new_bars: Iterable[Bar]) -> None:
        win = self._data.get(symbol)
        if win is None:
            win = BarWindow(maxlen=self._window_size, bars=deque(maxlen=self._window_size))
            self._data[symbol] = win

        # Update Memory
        existing = {self._normalize_ts(b.ts): i for i, b in enumerate(win.bars)}
        
        db_updates = []
        for b in new_bars:
            ts = self._normalize_ts(b.ts)
            normalized_bar = Bar(
                ts=ts,
                open=b.open,
                high=b.high,
                low=b.low,
                close=b.close,
                volume=b.volume,
                amount=b.amount,
            )
            
            # Prepare for DB
            db_updates.append((
                symbol,
                ts.isoformat(),
                float(b.open),
                float(b.high),
                float(b.low),
                float(b.close),
                float(getattr(b, "volume", 0.0) or 0.0),
                float(getattr(b, "amount", 0.0) or 0.0)
            ))

            if ts in existing:
                idx = existing[ts]
                tmp = list(win.bars)
                tmp[idx] = normalized_bar
                win.bars = deque(tmp, maxlen=win.maxlen)
                # Rebuild index if we modify in place
                existing = {self._normalize_ts(bb.ts): i for i, bb in enumerate(win.bars)}
            else:
                win.bars.append(normalized_bar)
                existing[ts] = len(win.bars) - 1

        # Ensure sorted
        win.bars = deque(sorted(win.bars, key=lambda x: x.ts), maxlen=win.maxlen)

        # Update SQLite
        if self._db_path and db_updates:
            with sqlite3.connect(self._db_path) as conn:
                conn.executemany("""
                    INSERT INTO bars (symbol, ts, open, high, low, close, volume, amount)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(symbol, ts) DO UPDATE SET
                        open=excluded.open,
                        high=excluded.high,
                        low=excluded.low,
                        close=excluded.close,
                        volume=excluded.volume,
                        amount=excluded.amount
                """, db_updates)

    def get_signals(self, symbol: str, limit: int = 2000, date: Optional[datetime] = None) -> List[Dict[str, object]]:
        """Load saved signals for the given symbol from SQLite.

        Returns rows ordered by ts ascending.
        """
        if not self._db_path:
            return []

        # Default to today's date in Beijing time
        if date is None:
            import datetime as dt

            beijing_now = datetime.now(dt.timezone(dt.timedelta(hours=8)))
            date = beijing_now.replace(tzinfo=None)

        date_prefix = date.date().isoformat()

        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT ts, signal_type, price, rule_reason, llm_action, llm_confidence, llm_reasons, t_share
                FROM signals
                WHERE symbol = ? AND ts LIKE ?
                ORDER BY ts ASC
                LIMIT ?
                """,
                (symbol, f"{date_prefix}%", int(limit)),
            )
            rows = cursor.fetchall()

        out: List[Dict[str, object]] = []
        for row in rows:
            out.append(
                {
                    "ts": row[0],
                    "signal_type": row[1],
                    "price": row[2],
                    "rule_reason": row[3],
                    "llm_action": row[4],
                    "llm_confidence": row[5],
                    "llm_reasons": row[6],
                    "t_share": row[7],
                }
            )
        return out

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
