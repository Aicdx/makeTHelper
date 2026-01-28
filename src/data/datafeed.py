from __future__ import annotations

import os
from typing import Callable, List, Tuple

from src.indicators.types import Bar


def fetch_bars(cfg: dict, symbol: str) -> Tuple[List[Bar], str]:
    """Fetch bars using primary source, fallback to secondary.

    Returns (bars, source_used).
    """

    ds = cfg.get("data_source", {})
    primary = (ds.get("primary") or "akshare").lower()
    fallback = (ds.get("fallback") or "akshare").lower()

    def _akshare() -> Tuple[List[Bar], str]:
        from src.data.datafeed_akshare import fetch_minute_bars as f

        bars, status = f(symbol)
        return bars, f"akshare:{status.provider}:{status.reason}"

    def _gateway() -> Tuple[List[Bar], str]:
        gw = ds.get("gateway", {})
        base_url = gw.get("base_url") or os.getenv("GATEWAY_BASE_URL", "")
        token_env = gw.get("token_env") or "GATEWAY_TOKEN"
        token = os.getenv(token_env, "")

        from src.data.datafeed_gateway import fetch_minute_bars as f

        bars, status = f(symbol=symbol, base_url=base_url, token=token)
        return bars, f"gateway:{status.reason}"

    sources: dict[str, Callable[[], Tuple[List[Bar], str]]] = {
        "akshare": _akshare,
        "gateway": _gateway,
    }

    for name in [primary, fallback]:
        fn = sources.get(name)
        if not fn:
            continue
        bars, tag = fn()
        if bars:
            return bars, tag

    return [], f"{primary}-> {fallback}:empty"

