from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from src.indicators.indicators import IndicatorSnapshot

ActionSide = Literal["BUY", "SELL"]


@dataclass(frozen=True)
class ScoreBreakdown:
    price_position: float
    macd_momentum: float
    volume: float
    ma_trend: float


@dataclass(frozen=True)
class AlertPolicyConfig:
    trading_hours_only: bool
    max_data_age_seconds: int
    lookback_bars: int

    max_buy: int
    max_sell: int
    enable_replacement: bool
    replace_min_improve_ratio: float

    near_extreme_ratio: float
    volume_multiplier_buy: float
    macd_hist_trend_bars: int

    score_weights: Dict[str, float]

    log_dir: str
    log_file_prefix: str


def load_alert_policy_cfg(cfg: Dict[str, Any]) -> AlertPolicyConfig:
    ap = cfg.get("alert_policy", {}) if isinstance(cfg, dict) else {}
    db = ap.get("daily_budget", {}) if isinstance(ap, dict) else {}
    lh = ap.get("low_high", {}) if isinstance(ap, dict) else {}
    sw = ap.get("score_weights", {}) if isinstance(ap, dict) else {}
    lg = ap.get("log", {}) if isinstance(ap, dict) else {}

    weights = {
        "price_position": float(sw.get("price_position", 0.45)),
        "macd_momentum": float(sw.get("macd_momentum", 0.30)),
        "volume": float(sw.get("volume", 0.15)),
        "ma_trend": float(sw.get("ma_trend", 0.10)),
    }

    return AlertPolicyConfig(
        trading_hours_only=bool(ap.get("trading_hours_only", True)),
        max_data_age_seconds=int(ap.get("max_data_age_seconds", 120)),
        lookback_bars=int(ap.get("lookback_bars", 20)),
        max_buy=int(db.get("max_buy", 2)),
        max_sell=int(db.get("max_sell", 2)),
        enable_replacement=bool(db.get("enable_replacement", True)),
        replace_min_improve_ratio=float(db.get("replace_min_improve_ratio", 0.005)),
        near_extreme_ratio=float(lh.get("near_extreme_ratio", 0.005)),
        volume_multiplier_buy=float(lh.get("volume_multiplier_buy", 1.2)),
        macd_hist_trend_bars=int(lh.get("macd_hist_trend_bars", 3)),
        score_weights=weights,
        log_dir=str(lg.get("dir", "logs")),
        log_file_prefix=str(lg.get("file_prefix", "trade_decisions")),
    )


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def score_low_high(
    side: ActionSide,
    closes: List[float],
    snap: IndicatorSnapshot,
    lookback: int,
    near_extreme_ratio: float,
    volume_multiplier_buy: float,
    hist_series: Optional[List[float]] = None,
) -> Tuple[float, ScoreBreakdown, Dict[str, Any]]:
    """低点/高点评分：返回(score 0~1, 分项, 额外特征)."""

    if not closes:
        bd = ScoreBreakdown(0.0, 0.0, 0.0, 0.0)
        return 0.0, bd, {"reason": "no_closes"}

    lb = max(5, int(lookback))
    series = closes[-lb:] if len(closes) >= lb else closes
    cur = float(series[-1])
    lo = float(min(series))
    hi = float(max(series))
    rng = max(1e-9, hi - lo)

    # 1) 价格在近期区间的位置
    pos = (cur - lo) / rng
    if side == "BUY":
        price_score = 1.0 - pos
        if cur <= lo * (1.0 + near_extreme_ratio):
            price_score = 1.0
    else:
        price_score = pos
        if cur >= hi * (1.0 - near_extreme_ratio):
            price_score = 1.0
    price_score = _clamp01(price_score)

    # 2) 动量/拐点（MACD hist）
    macd_score = 0.5
    if hist_series and len(hist_series) >= 3:
        a, b, c = float(hist_series[-3]), float(hist_series[-2]), float(hist_series[-1])
        if side == "BUY":
            if c > b > a and c < 0:
                macd_score = 0.8
            elif c > 0 and b <= 0:
                macd_score = 1.0
            elif c > b:
                macd_score = 0.6
        else:
            if c < b < a and c > 0:
                macd_score = 0.8
            elif c < 0 and b >= 0:
                macd_score = 1.0
            elif c < b:
                macd_score = 0.6
    else:
        # fallback: 仅基于 dif/dea 与 hist 符号
        if side == "BUY":
            if snap.dif >= snap.dea and snap.hist >= 0:
                macd_score = 0.75
            elif snap.hist > 0:
                macd_score = 0.6
        else:
            if snap.dif <= snap.dea and snap.hist <= 0:
                macd_score = 0.75
            elif snap.hist < 0:
                macd_score = 0.6
    macd_score = _clamp01(macd_score)

    # 3) 量能确认
    vol_score = 0.5
    if snap.vol is not None and snap.vma is not None and snap.vma > 0:
        ratio = float(snap.vol) / float(snap.vma)
        if side == "BUY":
            # BUY 更偏好放量（>= volume_multiplier_buy 倍附近达到满分）
            vm = max(1e-6, float(volume_multiplier_buy))
            vol_score = _clamp01(ratio / vm)
        else:
            # SELL 只要不太弱（>=均量接近满分）
            vol_score = _clamp01(ratio / 1.0)

    # 4) 趋势一致性
    ma_score = 0.5
    if snap.ma_trend is not None and snap.ma_trend > 0:
        if side == "BUY":
            ma_score = 0.7 if cur >= float(snap.ma_trend) else 0.3
        else:
            ma_score = 0.7 if cur <= float(snap.ma_trend) else 0.3

    bd = ScoreBreakdown(price_position=price_score, macd_momentum=macd_score, volume=vol_score, ma_trend=ma_score)

    features = {
        "lookback_used": len(series),
        "close": cur,
        "low": lo,
        "high": hi,
        "pos": pos,
        "dif": snap.dif,
        "dea": snap.dea,
        "hist": snap.hist,
        "vol": snap.vol,
        "vma": snap.vma,
        "ma_trend": snap.ma_trend,
        "near_extreme_ratio": near_extreme_ratio,
    }

    # score 的最终加权由调用方用 cfg.score_weights 计算
    return (
        (bd.price_position + bd.macd_momentum + bd.volume + bd.ma_trend) / 4.0,
        bd,
        features,
    )


class DecisionLogger:
    """追加写 JSONL，不覆盖历史文件。按天分文件。"""

    def __init__(self, log_dir: str, file_prefix: str):
        self._dir = Path(log_dir)
        self._prefix = file_prefix
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, day: str) -> Path:
        return self._dir / f"{self._prefix}_{day}.jsonl"

    def append(self, record: Dict[str, Any]) -> None:
        day = datetime.now().strftime("%Y-%m-%d")
        p = self._path(day)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


class DailyBudget:
    """每天 BUY/SELL 各保留 Top-N；满额后允许按比例阈值替换更差者。"""

    def __init__(
        self,
        max_buy: int,
        max_sell: int,
        enable_replacement: bool,
        replace_min_improve_ratio: float,
    ):
        self.max_buy = max(0, int(max_buy))
        self.max_sell = max(0, int(max_sell))
        self.enable_replacement = bool(enable_replacement)
        self.replace_min_improve_ratio = float(replace_min_improve_ratio)
        self._day = datetime.now().strftime("%Y-%m-%d")
        self._items: Dict[ActionSide, List[Dict[str, Any]]] = {"BUY": [], "SELL": []}

    def _roll_day_if_needed(self) -> None:
        day = datetime.now().strftime("%Y-%m-%d")
        if day != self._day:
            self._day = day
            self._items = {"BUY": [], "SELL": []}

    def consider(self, side: ActionSide, candidate: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        self._roll_day_if_needed()
        score = float(candidate.get("score", 0.0))
        items = self._items[side]
        limit = self.max_buy if side == "BUY" else self.max_sell

        if limit <= 0:
            return False, None

        if len(items) < limit:
            items.append(candidate)
            return True, None

        if not self.enable_replacement:
            return False, None

        worst = min(items, key=lambda x: float(x.get("score", 0.0)))
        worst_score = float(worst.get("score", 0.0))

        # 用比例阈值： (new - old) >= old * ratio
        improve_ratio = self.replace_min_improve_ratio
        if score - worst_score >= abs(worst_score) * improve_ratio:
            items.remove(worst)
            items.append(candidate)
            return True, worst

        return False, None

    def snapshot(self) -> Dict[str, Any]:
        self._roll_day_if_needed()
        return {
            "day": self._day,
            "buy": [{"symbol": x.get("symbol"), "score": x.get("score"), "bar_ts": x.get("bar_ts")} for x in self._items["BUY"]],
            "sell": [{"symbol": x.get("symbol"), "score": x.get("score"), "bar_ts": x.get("bar_ts")} for x in self._items["SELL"]],
        }

