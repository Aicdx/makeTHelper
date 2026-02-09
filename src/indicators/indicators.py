from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class IndicatorSnapshot:
    dif: float
    dea: float
    hist: float
    ma_trend: float | None
    vma: float | None
    vol: float | None
    vwap: float | None  # 分时均价线
    bias_vwap: float | None  # 股价偏离均价线的百分比


def compute_vwap(prices: List[float], volumes: List[float], amounts: List[float] | None = None) -> float | None:
    """计算分时均价线 (VWAP = 累计成交额 / 累计成交量)"""
    if not prices or not volumes or len(prices) != len(volumes):
        return None
    
    # 如果没有直接提供成交额，则用价格*成交量估算（注：A股分时通常取 均价=成交额/成交量）
    if amounts:
        total_amount = sum(amounts)
        total_volume = sum(volumes)
    else:
        total_amount = sum(p * v for p, v in zip(prices, volumes))
        total_volume = sum(volumes)
    
    if total_volume == 0:
        return None
    return total_amount / total_volume


def _ema(arr: np.ndarray, span: int) -> np.ndarray:
    if arr.size == 0:
        return arr
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(arr, dtype=float)
    out[0] = float(arr[0])
    for i in range(1, arr.size):
        out[i] = alpha * float(arr[i]) + (1.0 - alpha) * out[i - 1]
    return out


def compute_macd(closes: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = np.asarray(closes, dtype=float)
    if c.size == 0:
        return c, c, c
    ema_fast = _ema(c, fast)
    ema_slow = _ema(c, slow)
    dif = ema_fast - ema_slow
    dea = _ema(dif, signal)
    hist = (dif - dea) * 2.0
    return dif, dea, hist


def compute_sma(values: List[float], n: int) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return v
    if n <= 1:
        return v
    out = np.full_like(v, fill_value=np.nan, dtype=float)
    if v.size < n:
        return out
    csum = np.cumsum(v)
    out[n - 1 :] = (csum[n - 1 :] - np.concatenate(([0.0], csum[:-n]))) / float(n)
    return out


def latest_snapshot(
    closes: List[float],
    volumes: List[float] | None,
    amounts: List[float] | None,
    macd_fast: int,
    macd_slow: int,
    macd_signal: int,
    ma_trend_n: int,
    vma_n: int,
) -> IndicatorSnapshot | None:
    if not closes:
        return None

    dif, dea, hist = compute_macd(closes, macd_fast, macd_slow, macd_signal)

    ma_trend = None
    if ma_trend_n and ma_trend_n > 1:
        ma_arr = compute_sma(closes, ma_trend_n)
        if not np.isnan(ma_arr[-1]):
            ma_trend = float(ma_arr[-1])

    vma = None
    vol = None
    vwap = None
    bias_vwap = None
    
    if volumes:
        vol = float(volumes[-1])
        vma_arr = compute_sma(volumes, vma_n)
        if vma_n and vma_n > 1 and vma_arr.size and not np.isnan(vma_arr[-1]):
            vma = float(vma_arr[-1])
        
        # 计算 VWAP (基于全天累积)
        vwap = compute_vwap(closes, volumes, amounts)
        if vwap:
            bias_vwap = (closes[-1] - vwap) / vwap * 100

    return IndicatorSnapshot(
        dif=float(dif[-1]),
        dea=float(dea[-1]),
        hist=float(hist[-1]),
        ma_trend=ma_trend,
        vma=vma,
        vol=vol,
        vwap=vwap,
        bias_vwap=bias_vwap,
    )

