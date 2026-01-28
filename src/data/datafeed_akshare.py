from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import akshare as ak
import pandas as pd

from src.utils.symbols import to_akshare_code
from src.indicators.types import Bar


_logger = logging.getLogger(__name__)


@dataclass
class FetchStatus:
    ok: bool
    reason: str
    attempts: int
    provider: str


_fail_state: Dict[str, Tuple[int, datetime]] = {}  # symbol -> (fail_count, last_fail_at)


def _df_from_hist_min_em(symbol: str) -> pd.DataFrame:
    return ak.stock_zh_a_hist_min_em(symbol=to_akshare_code(symbol), period="1", adjust="")


def _df_from_pre_min_em(symbol: str) -> pd.DataFrame:
    # Daily intraday bars including pre-market, typically 1-minute granularity.
    # If the site is unstable for a symbol, this can sometimes work when hist_min fails.
    return ak.stock_zh_a_hist_pre_min_em(symbol=to_akshare_code(symbol))


def _should_skip(symbol: str) -> bool:
    state = _fail_state.get(symbol)
    if not state:
        return False
    fail_count, last_fail_at = state
    if fail_count >= 3:
        backoff_s = min(600, 60 * (2 ** (fail_count - 3)))
        return (datetime.now() - last_fail_at).total_seconds() < backoff_s
    return False


def _record_fail(symbol: str) -> int:
    fail_count, _ = _fail_state.get(symbol, (0, datetime.now()))
    fail_count += 1
    _fail_state[symbol] = (fail_count, datetime.now())
    return fail_count


def _record_ok(symbol: str) -> None:
    if symbol in _fail_state:
        del _fail_state[symbol]


def _normalize_df_to_bars(symbol: str, df: pd.DataFrame) -> List[Bar]:
    if df.empty:
        return []

    # Support both possible column sets
    # hist_min_em: 时间, 开盘, 收盘, 最高, 最低, 成交量, 成交额
    # pre_min_em : 时间, 开盘, 收盘, 最高, 最低, 成交量, 成交额 (usually same, but be defensive)
    required_cols = {"时间", "开盘", "收盘", "最高", "最低", "成交量"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"bad_columns: got={list(df.columns)}")

    bars: List[Bar] = []
    for _, row in df.iterrows():
        ts = pd.to_datetime(row["时间"])
        bars.append(
            Bar(
                ts=ts,
                open=float(row["开盘"]),
                high=float(row["最高"]),
                low=float(row["最低"]),
                close=float(row["收盘"]),
                volume=float(row["成交量"]),
                amount=float(row.get("成交额", 0.0)),
            )
        )
    bars.sort(key=lambda b: b.ts)
    return bars


def _try_provider(symbol: str, provider: str, max_retries: int) -> Tuple[List[Bar], str, int]:
    last_err: Exception | None = None

    for i in range(max_retries):
        try:
            if provider == "hist_min_em":
                df = _df_from_hist_min_em(symbol)
            elif provider == "pre_min_em":
                df = _df_from_pre_min_em(symbol)
            else:
                raise ValueError(f"unknown provider: {provider}")

            bars = _normalize_df_to_bars(symbol, df)
            if not bars:
                return [], "empty", i + 1
            return bars, "ok", i + 1
        except Exception as e:
            last_err = e
            sleep_s = (0.5 * (2**i)) + random.random() * 0.2
            if i == 0:
                _logger.warning(
                    "Fetch failed for %s via %s (attempt %d/%d): %s; retry in %.2fs",
                    symbol,
                    provider,
                    i + 1,
                    max_retries,
                    e,
                    sleep_s,
                )
            else:
                _logger.debug(
                    "Fetch failed for %s via %s (attempt %d/%d): %s; retry in %.2fs",
                    symbol,
                    provider,
                    i + 1,
                    max_retries,
                    e,
                    sleep_s,
                )
            time.sleep(sleep_s)

    return [], f"fetch_failed:{type(last_err).__name__}", max_retries


def fetch_minute_bars(symbol: str, max_retries: int = 3) -> Tuple[List[Bar], FetchStatus]:
    """Fetch 1-min bars with multi-provider fallback.

    Providers (in order):
    - hist_min_em (primary)
    - pre_min_em  (fallback)

    Also applies per-symbol circuit breaker to avoid log spam.
    """

    if _should_skip(symbol):
        return [], FetchStatus(ok=False, reason="circuit_breaker_skip", attempts=0, provider="-")

    providers = ["hist_min_em", "pre_min_em"]
    last_reason = ""
    last_attempts = 0

    for provider in providers:
        bars, reason, attempts = _try_provider(symbol, provider, max_retries=max_retries)
        last_reason = reason
        last_attempts = attempts
        if bars:
            _record_ok(symbol)
            return bars, FetchStatus(ok=True, reason=reason, attempts=attempts, provider=provider)

        # If provider returns empty, try next provider as well.
        # If provider fails, also try next provider.

    fail_count = _record_fail(symbol)
    if fail_count in (1, 3, 5, 8):
        _logger.error(
            "All providers failed for %s (fail_count=%d): %s",
            symbol,
            fail_count,
            last_reason,
        )

    return [], FetchStatus(ok=False, reason=last_reason or "fetch_failed", attempts=last_attempts, provider="all")
