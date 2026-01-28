from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd

from src.indicators.types import Bar


@dataclass
class FetchStatus:
    ok: bool
    reason: str
    attempts: int
    provider: str


def _parse_ts(ts: str) -> datetime:
    # Expect ISO8601 string (with timezone recommended)
    try:
        return pd.to_datetime(ts).to_pydatetime()
    except Exception:
        # fallback: naive parse
        return datetime.fromisoformat(ts)


def fetch_minute_bars(
    symbol: str,
    base_url: str,
    token: str,
    limit: int = 300,
    interval: str = "1m",
    timeout_s: int = 10,
) -> Tuple[List[Bar], FetchStatus]:
    """Fetch 1-min bars from your local gateway (FastAPI).

    The gateway accepts symbol like sh600519/sz300058.
    """

    if not base_url:
        return [], FetchStatus(ok=False, reason="base_url_missing", attempts=0, provider="gateway")
    if not token:
        return [], FetchStatus(ok=False, reason="token_missing", attempts=0, provider="gateway")

    q = urllib.parse.urlencode({"symbol": symbol, "limit": str(limit), "interval": interval})
    url = f"{base_url.rstrip('/')}/bars?{q}"

    req = urllib.request.Request(
        url,
        method="GET",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8")
        obj = json.loads(body)
    except Exception as e:
        return [], FetchStatus(ok=False, reason=f"http_error:{type(e).__name__}", attempts=1, provider="gateway")

    bars_obj = obj.get("bars", [])
    if not isinstance(bars_obj, list) or not bars_obj:
        return [], FetchStatus(ok=True, reason="empty", attempts=1, provider="gateway")

    bars: List[Bar] = []
    for row in bars_obj:
        try:
            bars.append(
                Bar(
                    ts=_parse_ts(str(row.get("ts"))),
                    open=float(row.get("open")),
                    high=float(row.get("high")),
                    low=float(row.get("low")),
                    close=float(row.get("close")),
                    volume=float(row.get("volume", 0.0)),
                    amount=float(row.get("amount", 0.0)),
                )
            )
        except Exception:
            continue

    bars.sort(key=lambda b: b.ts)
    return bars, FetchStatus(ok=True, reason="ok", attempts=1, provider="gateway")

