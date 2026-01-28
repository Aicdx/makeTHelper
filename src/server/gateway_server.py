from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query

from src.utils.ths_symbols import to_thsdk_code
from src.utils.dotenv_loader import load_env

try:
    from thsdk import THS
except Exception:  # pragma: no cover
    THS = None  # type: ignore


# Load env.local/.env once at startup so uvicorn process has GATEWAY_TOKEN/THS_* available.
load_env()

app = FastAPI(title="Market Gateway", version="0.1.0")


def _require_token(authorization: Optional[str] = Header(default=None)) -> None:
    token = os.getenv("GATEWAY_TOKEN")
    if not token:
        raise HTTPException(status_code=500, detail="GATEWAY_TOKEN not set")

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization: Bearer <token>")

    got = authorization.removeprefix("Bearer ").strip()
    if got != token:
        raise HTTPException(status_code=403, detail="Invalid token")


def _getf(row, *keys: str, default=None):
    for k in keys:
        v = row.get(k)
        if v is not None:
            return v
    return default


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "ts": datetime.now().isoformat(timespec="seconds"),
        "source": "thsdk" if THS is not None else "unavailable",
        "gateway_token_set": bool(os.getenv("GATEWAY_TOKEN")),
    }


@app.get("/bars")
def bars(
    symbol: str = Query(..., description="e.g. sz300058/sh600519 or USZA300058/USHA600519"),
    limit: int = Query(300, ge=1, le=2000),
    interval: str = Query("1m", description="1m/5m/15m/30m/60m/day"),
    _auth: None = Depends(_require_token),
) -> dict:
    if THS is None:
        raise HTTPException(status_code=500, detail="thsdk is not available in this environment")

    ths_symbol = to_thsdk_code(symbol)

    try:
        with THS() as ths:
            resp = ths.klines(ths_symbol, count=int(limit), interval=interval)
            if not resp:
                raise HTTPException(status_code=502, detail=f"thsdk error: {resp.error}")

            df = resp.df
            if df.empty:
                return {"symbol": ths_symbol, "bars": []}

            out = []
            for _, row in df.iterrows():
                ts = _getf(row, "时间")
                ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)

                # thsdk columns are typically: 开盘价/收盘价/最高价/最低价/成交量/总金额
                o = float(_getf(row, "开盘", "开盘价"))
                c = float(_getf(row, "收盘", "收盘价"))
                h = float(_getf(row, "最高", "最高价"))
                l = float(_getf(row, "最低", "最低价"))
                vol = float(_getf(row, "成交量", default=0.0) or 0.0)
                amt = float(_getf(row, "成交额", "总金额", default=0.0) or 0.0)

                out.append(
                    {
                        "ts": ts_str,
                        "open": o,
                        "high": h,
                        "low": l,
                        "close": c,
                        "volume": vol,
                        "amount": amt,
                    }
                )

            return {"symbol": ths_symbol, "bars": out}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"gateway parse error: {type(e).__name__}: {e}")
