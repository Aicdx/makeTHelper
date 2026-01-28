from __future__ import annotations

import re


_SYMBOL_RE = re.compile(r"^(sh|sz)(\d{6})$")


def normalize_symbol(symbol: str) -> str:
    s = symbol.strip().lower()
    m = _SYMBOL_RE.match(s)
    if not m:
        raise ValueError(f"Invalid symbol '{symbol}'. Expected like sh600519 or sz000001")
    return s


def to_akshare_code(symbol: str) -> str:
    """Return 6-digit code used by many akshare endpoints."""
    s = normalize_symbol(symbol)
    return s[2:]


def market_of(symbol: str) -> str:
    s = normalize_symbol(symbol)
    return s[:2]

