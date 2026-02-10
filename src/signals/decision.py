from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
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
    # 增加开盘模式相关字段
    prev_close: Optional[float] = None
    open_price: Optional[float] = None
    # 增加盈亏分析字段
    last_op_price: Optional[float] = None


def _summarize_series(values: List[float], n: int = 20) -> List[float]:
    if not values:
        return []
    return [float(v) for v in values[-n:]]


def build_llm_payload(ctx: DecisionContext, position_state: str, t_share: float) -> Dict[str, Any]:
    ind = ctx.indicators

    prev_close = float(ctx.prev_close) if ctx.prev_close is not None else None
    open_price = float(ctx.open_price) if ctx.open_price is not None else None
    gap_pct = (
        ((open_price - prev_close) / prev_close)
        if (prev_close is not None and open_price is not None and prev_close != 0.0)
        else None
    )
    t = ctx.ts.time()
    session_phase = "OPENING" if (time(9, 30) <= t < time(10, 0)) else "NORMAL"

    # 计算盈亏情况
    last_op_price = float(ctx.last_op_price) if ctx.last_op_price is not None else None
    profit_ratio = (
        (float(ctx.close) - last_op_price) / last_op_price
        if (last_op_price is not None and last_op_price > 0)
        else None
    )

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
            "vwap": float(ind.vwap) if ind.vwap is not None else None,
            "bias_vwap": float(ind.bias_vwap) if ind.bias_vwap is not None else None,
        },
        "recent": {
            "timestamps": ctx.recent_timestamps[-20:],
            "closes": _summarize_series(ctx.recent_closes, 20),
            "volumes": _summarize_series(ctx.recent_volumes, 20),
        },
        "rule_event": ctx.rule_event,
        "session": {
            "phase": session_phase,
            "prev_close": prev_close,
            "open_price": open_price,
            "gap_pct": gap_pct,
        },
        "portfolio": {
            "last_op_price": last_op_price,
            "profit_ratio": profit_ratio,
        },
        "trend_context": {
            "ma_trend": float(ind.ma_trend) if ind.ma_trend is not None else None,
            "close": float(ctx.close),
            "above_ma_trend": (float(ctx.close) >= float(ind.ma_trend)) if ind.ma_trend is not None else None,
            "ma_distance_ratio": (
                (float(ctx.close) - float(ind.ma_trend)) / float(ind.ma_trend)
                if ind.ma_trend is not None and float(ind.ma_trend) != 0.0
                else None
            ),
        },
        "trading_context": {
            "base_share": 1.0,
            "t_share": float(t_share),
            "t_step": 0.5,
            "t_share_min": -1.0,
            "t_share_max": 1.0,
            "position_state": position_state,
        },
        "constraints": {
            "market": "A股",
            "timeframe": "1min",
            "mode": "只告警，不下单",
            "allowed_actions": ["BUY_BACK", "SELL_PART", "HOLD_POSITION", "CANCEL_PLAN"],
        },
    }
    return payload


def llm_decision(ctx: DecisionContext, position_state: str, t_share: float) -> LLMDecision:
    payload = build_llm_payload(ctx, position_state=position_state, t_share=t_share)
    return llm_decide(payload, ts=ctx.ts)

