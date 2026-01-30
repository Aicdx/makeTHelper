from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
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


class SignalEngine:
    def __init__(
        self,
        confirm_bars: int = 2,
        cooldown_minutes: int = 15,
        volume_multiplier: float = 1.2,
        enable_trend_filter: bool = True,
    ):
        self.confirm_bars = confirm_bars
        self.cooldown = timedelta(minutes=cooldown_minutes)
        self.volume_multiplier = volume_multiplier
        self.enable_trend_filter = enable_trend_filter
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

    def evaluate(
        self,
        symbol: str,
        ts: datetime,
        close: float,
        ind: IndicatorSnapshot,
    ) -> SignalEvent | None:
        st = self._get_state(symbol)


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

