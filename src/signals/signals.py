from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from enum import Enum
from typing import Dict, Optional

from src.indicators.indicators import IndicatorSnapshot


class SignalType(str, Enum):
    BUY_BACK = "BUY_BACK"
    SELL_PART = "SELL_PART"
    HOLD_POSITION = "HOLD_POSITION"
    CANCEL_PLAN = "CANCEL_PLAN"
    INFO = "INFO"


@dataclass(frozen=True)
class SignalEvent:
    symbol: str
    ts: datetime
    signal: SignalType
    reason: str
    dif: float
    dea: float
    hist: float
    vol: float | None
    vma: float | None
    close: float


@dataclass
class SymbolState:
    last_emit_at: Dict[SignalType, datetime]
    last_relation: Optional[int] = None  # 1: dif>dea, -1: dif<dea
    pending_cross: Optional[SignalType] = None
    pending_since: Optional[datetime] = None
    pending_count: int = 0

    # opening-mode specific
    last_opening_relation: Optional[int] = None  # 1: close>prev_close, -1: close<prev_close
    pending_opening_cross: Optional[SignalType] = None
    pending_opening_since: Optional[datetime] = None
    pending_opening_count: int = 0


class SignalEngine:
    def __init__(
        self,
        confirm_bars: int = 2,
        cooldown_minutes: int = 15,
        volume_multiplier: float = 1.2,
        enable_trend_filter: bool = True,
        opening_confirm_bars: int = 1,
        opening_cooldown_minutes: int = 5,
        opening_gap_threshold: float = 0.006,
    ):
        self.confirm_bars = confirm_bars
        self.cooldown = timedelta(minutes=cooldown_minutes)
        self.volume_multiplier = volume_multiplier
        self.enable_trend_filter = enable_trend_filter

        self.opening_confirm_bars = max(1, opening_confirm_bars)
        self.opening_cooldown = timedelta(minutes=opening_cooldown_minutes)
        self.opening_gap_threshold = max(0.0, opening_gap_threshold)

        self._state: Dict[str, SymbolState] = {}

    def _get_state(self, symbol: str) -> SymbolState:
        st = self._state.get(symbol)
        if st is None:
            st = SymbolState(last_emit_at={})
            self._state[symbol] = st
        return st

    def _cooldown_ok(self, st: SymbolState, sig: SignalType, now: datetime) -> bool:
        last = st.last_emit_at.get(sig)
        if last is None:
            return True
        return (now - last) >= self.cooldown

    def _opening_cooldown_ok(self, st: SymbolState, sig: SignalType, now: datetime) -> bool:
        last = st.last_emit_at.get(sig)
        if last is None:
            return True
        return (now - last) >= self.opening_cooldown

    @staticmethod
    def _is_opening_session(ts: datetime) -> bool:
        t = ts.time()
        return time(9, 30) <= t < time(10, 0)

    def _evaluate_opening(
        self,
        st: SymbolState,
        symbol: str,
        ts: datetime,
        close: float,
        ind: IndicatorSnapshot,
        *,
        prev_close: float,
        open_price: float | None,
    ) -> SignalEvent | None:
        if prev_close <= 0:
            return None

        gap_ref = open_price if (open_price is not None and open_price > 0) else close
        gap_pct = (gap_ref - prev_close) / prev_close
        
        # 即使 gap 小，如果偏离度 bias 够大，也进入开盘模式
        bias = ind.bias_vwap if ind.bias_vwap is not None else 0.0
        
        # 如果既没跳空，也没偏离，则跳过
        if abs(gap_pct) < self.opening_gap_threshold and abs(bias) < 1.0:
            return None

        relation = 1 if close > prev_close else (-1 if close < prev_close else 0)
        
        # 核心逻辑 A: 穿越昨收 (补缺口)
        cross: SignalType | None = None
        if st.last_opening_relation is not None and abs(gap_pct) >= self.opening_gap_threshold:
            if gap_pct < 0 and st.last_opening_relation <= 0 and relation == 1:
                cross = SignalType.BUY_BACK
            elif gap_pct > 0 and st.last_opening_relation >= 0 and relation == -1:
                cross = SignalType.SELL_PART

        st.last_opening_relation = relation

        # 核心逻辑 B: 极端乖离率回归 (针对平开急跌或高开急拉)
        if cross is None:
            # 趋势判定：如果价格在 MA60 下方，说明处于弱势，触发门槛需更严格
            is_weak_trend = (ind.ma_trend is not None and close < ind.ma_trend)
            bias_threshold = -2.5 if is_weak_trend else -1.5
            
            # bias < bias_threshold 通常意味着超卖，配合 hist 翻红（或空头缩短）触发
            if bias < bias_threshold and ind.hist > -0.01: # 稍微拐头即触发
                 cross = SignalType.BUY_BACK
            elif bias > 1.5 and ind.hist < 0.01: # 稍微滞涨即触发
                 cross = SignalType.SELL_PART

        if cross is None:
            if st.pending_opening_cross is not None and relation == 0:
                st.pending_opening_cross = None
                st.pending_opening_since = None
                st.pending_opening_count = 0
            return None

        if st.pending_opening_cross != cross:
            st.pending_opening_cross = cross
            st.pending_opening_since = ts
            st.pending_opening_count = 1
        else:
            st.pending_opening_count += 1

        if st.pending_opening_count < self.opening_confirm_bars:
            return None

        if not self._opening_cooldown_ok(st, cross, ts):
            return None

        st.last_emit_at[cross] = ts
        st.pending_opening_cross = None
        st.pending_opening_since = None
        st.pending_opening_count = 0

        direction = "低开回补" if gap_pct < 0 else "高开回落"
        reason = f"开盘模式({direction})触发: gap={gap_pct*100:.2f}%, close穿越昨收={prev_close:.2f}"
        if ind.vma is not None and ind.vol is not None:
            reason += f"; vol={ind.vol:.0f}, vma={ind.vma:.0f}"

        return SignalEvent(
            symbol=symbol,
            ts=ts,
            signal=cross,
            reason=reason,
            dif=ind.dif,
            dea=ind.dea,
            hist=ind.hist,
            vol=ind.vol,
            vma=ind.vma,
            close=close,
        )

    def evaluate(
        self,
        symbol: str,
        ts: datetime,
        close: float,
        ind: IndicatorSnapshot,
        *,
        prev_close: float | None = None,
        open_price: float | None = None,
    ) -> SignalEvent | None:
        st = self._get_state(symbol)

        # Opening mode: 9:30-10:00 优先使用“昨收锚点 + gap”的快触发
        if self._is_opening_session(ts) and prev_close is not None:
            opening_ev = self._evaluate_opening(
                st,
                symbol,
                ts,
                close,
                ind,
                prev_close=prev_close,
                open_price=open_price,
            )
            if opening_ev is not None:
                return opening_ev

        relation = 1 if ind.dif > ind.dea else (-1 if ind.dif < ind.dea else 0)
        if st.last_relation is None:
            st.last_relation = relation
            return None

        # Detect crossing
        cross: SignalType | None = None
        if st.last_relation <= 0 and relation == 1:
            cross = SignalType.BUY_BACK
        elif st.last_relation >= 0 and relation == -1:
            cross = SignalType.SELL_PART

        st.last_relation = relation

        if cross is None:
            # reset pending if relation invalidates
            if st.pending_cross is not None and relation == 0:
                st.pending_cross = None
                st.pending_since = None
                st.pending_count = 0
            return None

        # Start or continue confirmation
        if st.pending_cross != cross:
            st.pending_cross = cross
            st.pending_since = ts
            st.pending_count = 1
        else:
            st.pending_count += 1

        if st.pending_count < self.confirm_bars:
            return None

        # Confirmed cross - apply cooldown & volume filter
        if not self._cooldown_ok(st, cross, ts):
            return None

        if cross == SignalType.BUY_BACK:
            if ind.vma is not None and ind.vol is not None:
                if ind.vol < ind.vma * self.volume_multiplier:
                    # weak volume: emit INFO at most
                    if self._cooldown_ok(st, SignalType.INFO, ts):
                        st.last_emit_at[SignalType.INFO] = ts
                        return SignalEvent(
                            symbol=symbol,
                            ts=ts,
                            signal=SignalType.INFO,
                            reason=f"金叉确认但未放量: vol={ind.vol:.0f} < vma*{self.volume_multiplier:.2f} ({ind.vma * self.volume_multiplier:.0f})",
                            dif=ind.dif,
                            dea=ind.dea,
                            hist=ind.hist,
                            vol=ind.vol,
                            vma=ind.vma,
                            close=close,
                        )
                    return None

        st.last_emit_at[cross] = ts
        st.pending_cross = None
        st.pending_since = None
        st.pending_count = 0

        reason = "MACD金叉确认" if cross == SignalType.BUY_BACK else "MACD死叉确认"
        if ind.vma is not None and ind.vol is not None:
            reason += f"; vol={ind.vol:.0f}, vma={ind.vma:.0f}"

        return SignalEvent(
            symbol=symbol,
            ts=ts,
            signal=cross,
            reason=reason,
            dif=ind.dif,
            dea=ind.dea,
            hist=ind.hist,
            vol=ind.vol,
            vma=ind.vma,
            close=close,
        )
