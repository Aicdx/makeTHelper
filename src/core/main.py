from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta

import yaml

from src.utils.alerter import emit
from src.data.bar_store import BarStore
from src.data.datafeed import fetch_bars
from src.signals.decision import DecisionContext, llm_decision
from src.indicators.indicators import latest_snapshot
from src.ai.llm_agent import LLMDecision
from src.signals.signals import SignalEngine, SignalType
from src.utils.symbols import normalize_symbol


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _is_trading_time_cn(now: datetime) -> bool:
    # Simple trading session check in local time (WSL uses system time)
    hm = now.hour * 60 + now.minute
    morning = 9 * 60 + 30 <= hm <= 11 * 60 + 30
    afternoon = 13 * 60 <= hm <= 15 * 60
    return morning or afternoon


def main() -> None:
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    _setup_logging(cfg.get("logging", {}).get("level", "INFO"))

    symbols = [normalize_symbol(s) for s in cfg.get("symbols", [])]
    poll_interval = int(cfg.get("poll_interval_seconds", 15))
    window_size = int(cfg.get("window_size", 600))

    ind_cfg = cfg.get("indicators", {})
    macd_cfg = ind_cfg.get("macd", {})
    macd_fast = int(macd_cfg.get("fast", 12))
    macd_slow = int(macd_cfg.get("slow", 26))
    macd_signal = int(macd_cfg.get("signal", 9))
    ma_trend_n = int(ind_cfg.get("ma_trend", 60))
    vma_n = int(ind_cfg.get("vma", 20))

    strat_cfg = cfg.get("strategy", {})
    engine = SignalEngine(
        confirm_bars=int(strat_cfg.get("confirm_bars", 2)),
        cooldown_minutes=int(strat_cfg.get("cooldown_minutes", 15)),
        volume_multiplier=float(strat_cfg.get("volume_multiplier", 1.2)),
        enable_trend_filter=bool(strat_cfg.get("enable_trend_filter", True)),
    )

    llm_cfg = cfg.get("llm_agent", {})
    llm_enabled = bool(llm_cfg.get("enabled", False))
    llm_min_conf = float(llm_cfg.get("min_confidence", 0.6))
    llm_throttle = timedelta(minutes=int(llm_cfg.get("throttle_minutes", 5)))

    store = BarStore(window_size=window_size)

    llm_last_call_at: dict[str, datetime] = {}

    log = logging.getLogger(__name__)
    log.info(f"Monitoring symbols: {symbols}")
    log.info(
        "LLM: enabled=%s model=%s base=%s",
        llm_enabled,
        os.getenv("LLM_MODEL", llm_cfg.get("model", "")),
        os.getenv("LLM_API_BASE", ""),
    )

    while True:
        now = datetime.now()
        if not _is_trading_time_cn(now):
            time.sleep(min(poll_interval, 30))
            continue

        for sym in symbols:
            bars, source_tag = fetch_bars(cfg, sym)
            if not bars:
                log.info("heartbeat %s: no bars fetched (source=%s)", sym, source_tag)
                continue

            store.upsert_bars(sym, bars)
            win = store.get_window(sym, 300)
            if len(win) < max(30, vma_n + 5, ma_trend_n + 5):
                log.info("heartbeat %s: insufficient window (%d)", sym, len(win))
                continue

            closes = [b.close for b in win]
            vols = [b.volume for b in win]
            snap = latest_snapshot(
                closes=closes,
                volumes=vols,
                macd_fast=macd_fast,
                macd_slow=macd_slow,
                macd_signal=macd_signal,
                ma_trend_n=ma_trend_n,
                vma_n=vma_n,
            )
            if snap is None:
                log.info("heartbeat %s: indicator snapshot is None", sym)
                continue

            latest_bar = win[-1]

            log.info(
                "heartbeat %s: ts=%s close=%.3f dif=%.4f dea=%.4f hist=%.4f vol=%s vma=%s ma=%s source=%s",
                sym,
                latest_bar.ts.strftime("%Y-%m-%d %H:%M"),
                latest_bar.close,
                snap.dif,
                snap.dea,
                snap.hist,
                f"{snap.vol:.0f}" if snap.vol is not None else "NA",
                f"{snap.vma:.0f}" if snap.vma is not None else "NA",
                f"{snap.ma_trend:.3f}" if snap.ma_trend is not None else "NA",
                source_tag,
            )

            event = engine.evaluate(
                symbol=sym,
                ts=latest_bar.ts,
                close=latest_bar.close,
                ind=snap,
            )

            if event is None:
                continue

            if llm_enabled and event.signal in {SignalType.BUY_T, SignalType.SELL_T, SignalType.DISABLE_T, SignalType.INFO}:
                last_call = llm_last_call_at.get(sym)
                if last_call is None or (latest_bar.ts - last_call) >= llm_throttle:
                    try:
                        ctx = DecisionContext(
                            symbol=sym,
                            ts=latest_bar.ts,
                            close=latest_bar.close,
                            indicators=snap,
                            recent_closes=closes,
                            recent_volumes=vols,
                            recent_timestamps=[b.ts.strftime("%Y-%m-%d %H:%M") for b in win],
                            rule_event={
                                "signal": event.signal.value,
                                "reason": event.reason,
                                "data_source": source_tag,
                            },
                        )
                        decision: LLMDecision = llm_decision(ctx)
                        llm_last_call_at[sym] = latest_bar.ts

                        if decision.confidence >= llm_min_conf and decision.action in {"BUY_T", "SELL_T", "DISABLE_T", "HOLD"}:
                            emit(
                                event.__class__(
                                    symbol=sym,
                                    ts=latest_bar.ts,
                                    signal=SignalType(decision.action)
                                    if decision.action in {"BUY_T", "SELL_T", "DISABLE_T"}
                                    else SignalType.INFO,
                                    reason=f"LLM action={decision.action} conf={decision.confidence:.2f}; reasons="
                                    + "; ".join(decision.reasons[:6]),
                                    dif=event.dif,
                                    dea=event.dea,
                                    hist=event.hist,
                                    vol=event.vol,
                                    vma=event.vma,
                                    close=event.close,
                                )
                            )
                        else:
                            emit(event)
                    except Exception as e:
                        log.error("LLM decision failed for %s: %s", sym, e)
                        emit(event)
                else:
                    emit(event)
            else:
                emit(event)

        time.sleep(poll_interval)


if __name__ == "__main__":
    main()
