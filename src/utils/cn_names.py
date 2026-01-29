from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Optional

from src.utils.symbols import normalize_symbol, to_akshare_code


@lru_cache(maxsize=4096)
def _load_local_map() -> dict[str, str]:
    """Load optional local mapping file: src/utils/cn_names.csv

    CSV format:
      symbol,name
      sz300986,xxx
      sh600519,贵州茅台

    If the file doesn't exist, returns empty map.
    """

    p = Path(__file__).with_name("cn_names.csv")
    if not p.exists():
        return {}

    out: dict[str, str] = {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sym = row.get("symbol")
                name = row.get("name")
                if not sym or not name:
                    continue
                try:
                    out[normalize_symbol(sym)] = str(name).strip()
                except Exception:
                    continue
    except Exception:
        return {}

    return out


def get_cn_name(symbol: str) -> Optional[str]:
    """Get Chinese name for symbol.

    Priority:
    1) local file mapping (cn_names.csv)
    2) akshare spot list (best effort, cached)
    """

    sym = normalize_symbol(symbol)

    local = _load_local_map().get(sym)
    if local:
        return local

    return _akshare_name(sym)


@lru_cache(maxsize=4096)
def _akshare_name(symbol: str) -> Optional[str]:
    try:
        import akshare as ak
    except Exception:
        return None

    code = to_akshare_code(symbol)

    # Best effort: try common spot endpoint
    try:
        df = ak.stock_zh_a_spot_em()
        # columns often include: 代码, 名称
        if "代码" not in df.columns or "名称" not in df.columns:
            return None
        row = df[df["代码"] == code]
        if row.empty:
            return None
        name = row.iloc[0]["名称"]
        if name is None:
            return None
        name = str(name).strip()
        return name if name else None
    except Exception:
        return None

