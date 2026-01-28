from __future__ import annotations

import re


_A_RE = re.compile(r"^(sh|sz)(\d{6})$", re.IGNORECASE)
_THS_RE = re.compile(r"^(USHA|USZA)(\d{6})$", re.IGNORECASE)


def to_thsdk_code(symbol: str) -> str:
    """Normalize input to thsdk code.

    Accepts:
    - sh600519 / sz300058 (preferred in this project)
    - USHA600519 / USZA300058 (thsdk native)
    """

    s = symbol.strip()
    m = _THS_RE.match(s)
    if m:
        return (m.group(1).upper() + m.group(2))

    m = _A_RE.match(s)
    if m:
        prefix = m.group(1).lower()
        code = m.group(2)
        return ("USHA" if prefix == "sh" else "USZA") + code

    raise ValueError(f"Invalid symbol '{symbol}'. Expected sh600519/sz300058 or USHA600519/USZA300058")

