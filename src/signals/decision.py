from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.indicators.indicators import IndicatorSnapshot
from src.ai.llm_agent import LLMDecision, decide as llm_decide


@dataclass(frozen=True)
class DecisionContext:
    symbol: str
    ts: datetime
    close: float
    indicators: IndicatorSnapshot
    recent_closes: List[float]
    recent_volumes: List[float]
    recent_timestamps: List[str]
    rule_event: Optional[Dict[str, Any]] = None


def _summarize_series(values: List[float], n: int = 20) -> List[float]:
    if not values:
        return []
    return [float(v) for v in values[-n:]]


def build_llm_payload(ctx: DecisionContext) -> Dict[str, Any]:
    ind = ctx.indicators
    payload: Dict[str, Any] = {
        "symbol": ctx.symbol,
        "ts": ctx.ts.isoformat(timespec="seconds"),
        "close": float(ctx.close),
        "indicators": {
            "dif": float(ind.dif),
            "dea": float(ind.dea),
            "hist": float(ind.hist),
            "ma_trend": float(ind.ma_trend) if ind.ma_trend is not None else None,
            "vma": float(ind.vma) if ind.vma is not None else None,
            "vol": float(ind.vol) if ind.vol is not None else None,
        },
        "recent": {
            "timestamps": ctx.recent_timestamps[-20:],
            "closes": _summarize_series(ctx.recent_closes, 20),
            "volumes": _summarize_series(ctx.recent_volumes, 20),
        },
        "rule_event": ctx.rule_event,
        "constraints": {
            "market": "A股",
            "timeframe": "1min",
            "mode": "只告警，不下单",
            "allowed_actions": ["BUY_T", "SELL_T", "HOLD", "DISABLE_T"],
        },
    }
    return payload


def llm_decision(ctx: DecisionContext) -> LLMDecision:
    payload = build_llm_payload(ctx)
    return llm_decide(payload)

