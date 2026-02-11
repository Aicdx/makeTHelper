from __future__ import annotations

import logging
import os
import sys
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
from src.utils.dotenv_loader import load_env
from src.utils.alert_policy import DailyBudget, DecisionLogger, load_alert_policy_cfg, score_low_high
from src.utils.cn_names import get_cn_name
from src.server.shared_state import data_bus
from src.server.monitor_server import start_monitor_server
from src.server import shared_state


def _setup_logging(level: str) -> None:
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        handlers=[handler],
        force=True,
    )


def _is_trading_time_cn(now: datetime) -> bool:
    hm = now.hour * 60 + now.minute
    morning = 9 * 60 + 30 <= hm <= 11 * 60 + 30
    afternoon = 13 * 60 <= hm <= 15 * 60
    return morning or afternoon


def main() -> None:
    load_env()

    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    _setup_logging(cfg.get("logging", {}).get("level", "INFO"))

    # Start the monitor server in a background thread
    monitor_cfg = cfg.get("monitor_server", {})
    if monitor_cfg.get("enabled", True):
        start_monitor_server(
            host=monitor_cfg.get("host", "127.0.0.1"),
            port=monitor_cfg.get("port", 18080),
        )

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
        confirm_bars=int(strat_cfg.get("confirm_bars", 1)),
        cooldown_minutes=int(strat_cfg.get("cooldown_minutes", 5)),
        volume_multiplier=float(strat_cfg.get("volume_multiplier", 0.8)),
        enable_trend_filter=bool(strat_cfg.get("enable_trend_filter", False)),
        opening_gap_threshold=float(strat_cfg.get("opening_gap_threshold", 0.003)),
        opening_confirm_bars=int(strat_cfg.get("opening_confirm_bars", 1)),
        opening_cooldown_minutes=int(strat_cfg.get("opening_cooldown_minutes", 15)),
    )

    llm_cfg = cfg.get("llm_agent", {})
    llm_enabled = bool(llm_cfg.get("enabled", False))
    llm_min_conf = float(llm_cfg.get("min_confidence", 0.6))
    llm_throttle = timedelta(minutes=int(llm_cfg.get("throttle_minutes", 5)))

    store = BarStore(window_size=window_size, db_path=os.getenv("BARSTORE_DB_PATH", "data/bars.db"))
    shared_state.bar_store = store

    llm_last_call_at: dict[str, datetime] = {}
    last_buy_price_by_symbol: dict[str, float] = {sym: 0.0 for sym in symbols}

    position_state_by_symbol: dict[str, str] = {sym: "HOLDING_STOCK" for sym in symbols}
    t_share_by_symbol: dict[str, float] = {sym: 0.0 for sym in symbols}

    test_llm_done: dict[str, bool] = {}

    log = logging.getLogger(__name__)
    log.info(f"Monitoring symbols: {symbols}")
    log.info(
        "LLM: enabled=%s model=%s base=%s",
        llm_enabled,
        os.getenv("LLM_MODEL", llm_cfg.get("model", "")),
        os.getenv("LLM_API_BASE", ""),
    )

    agent_cfg = cfg.get("agent", {})
    test_mode = bool(agent_cfg.get("test_mode", False))
    test_force_llm = bool(agent_cfg.get("test_force_llm", False))
    heartbeat_every_loops = int(agent_cfg.get("heartbeat_every_loops", 1))

    policy = load_alert_policy_cfg(cfg)
    decision_logger = DecisionLogger(log_dir=policy.log_dir, file_prefix=policy.log_file_prefix)
    daily_budget = DailyBudget(
        max_buy=policy.max_buy,
        max_sell=policy.max_sell,
        enable_replacement=policy.enable_replacement,
        replace_min_improve_ratio=policy.replace_min_improve_ratio,
    )

    if test_mode:
        log.warning("Test mode enabled: will run outside trading hours; test_force_llm=%s", test_force_llm)

    # --- 启动补齐逻辑 (Warmup) ---
    log.info("Starting warmup: fetching initial bars for %d symbols...", len(symbols))
    for sym in symbols:
        bars, source_tag = fetch_bars(cfg, sym)
        if bars:
            store.upsert_bars(sym, bars)
            log.info("Warmup %s: loaded %d bars from %s", sym, len(bars), source_tag)
        else:
            log.warning("Warmup %s: failed to fetch initial bars", sym)
    # --------------------------

    loop_i = 0
    while True:
        loop_i += 1
        now = datetime.now()
        if not test_mode and not _is_trading_time_cn(now):
            time.sleep(min(poll_interval, 30))
            continue

        if test_mode and (loop_i % max(1, heartbeat_every_loops) == 0):
            log.info("[TEST] loop=%d now=%s symbols=%d", loop_i, now.strftime("%Y-%m-%d %H:%M:%S"), len(symbols))

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
            amounts = [getattr(b, "amount", b.close * b.volume) for b in win]
            snap = latest_snapshot(
                closes=closes,
                volumes=vols,
                amounts=amounts,
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
            # 获取今日开盘价和昨收价（简单处理：从当前窗口中寻找今日第一根K线的open作为open_price）
            # 昨收价建议从之前的历史数据或配置中获取，这里我们尝试从win中找跨天分界
            curr_day = latest_bar.ts.date()
            today_open = None
            prev_close_val = None
            
            for i in range(len(win)):
                if win[i].ts.date() == curr_day:
                    if today_open is None:
                        today_open = win[i].open
                        if i > 0:
                            prev_close_val = win[i-1].close
                    break
            
            cn_name_for_bus = get_cn_name(sym) or ""
            data_bus.update_snapshot(
                sym,
                name=cn_name_for_bus,
                ts=latest_bar.ts,
                close=float(latest_bar.close),
                indicators=snap,
                position_state=position_state_by_symbol.get(sym, "HOLDING_STOCK"),
                t_share=float(t_share_by_symbol.get(sym, 0.0)),
            )
            data_bus.add_close_point(sym, latest_bar.ts, float(latest_bar.close))

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
                prev_close=prev_close_val,
                open_price=today_open,
            )

            if test_mode and test_force_llm and not test_llm_done.get(sym, False):
                from src.signals.signals import SignalEvent

                event = SignalEvent(
                    symbol=sym,
                    ts=latest_bar.ts,
                    signal=SignalType.INFO,
                    reason="TEST_FORCE_LLM_ONCE",
                    dif=snap.dif,
                    dea=snap.dea,
                    hist=snap.hist,
                    vol=snap.vol,
                    vma=snap.vma,
                    close=latest_bar.close,
                )
                test_llm_done[sym] = True
                log.warning(
                    "[TEST] first data ready -> forcing ONE LLM decision for %s at %s",
                    sym,
                    latest_bar.ts.strftime("%Y-%m-%d %H:%M"),
                )

            if event is None:
                continue

            if llm_enabled and event.signal in {SignalType.BUY_BACK, SignalType.SELL_PART, SignalType.CANCEL_PLAN, SignalType.HOLD_POSITION, SignalType.INFO}:
                last_call = llm_last_call_at.get(sym)
                bypass_throttle = test_mode and event.reason == "TEST_FORCE_LLM_ONCE"
                if bypass_throttle:
                    log.info("[TEST] bypass throttle for %s (first forced decision)", sym)

                if bypass_throttle or last_call is None or (latest_bar.ts - last_call) >= llm_throttle:
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
                                "trend_filter": {
                                    "enabled": bool(strat_cfg.get("enable_trend_filter", True)),
                                    "ma_trend": float(snap.ma_trend) if snap.ma_trend is not None else None,
                                    "close": float(latest_bar.close),
                                    "above_ma_trend": (float(latest_bar.close) >= float(snap.ma_trend)) if snap.ma_trend is not None else None,
                                },
                            },
                            prev_close=prev_close_val,
                            open_price=today_open,
                            last_op_price=last_buy_price_by_symbol.get(sym),
                        )
                        position_state = position_state_by_symbol.get(sym, "HOLDING_STOCK")
                        t_share = float(t_share_by_symbol.get(sym, 0.0))
                        decision: LLMDecision = llm_decision(ctx, position_state=position_state, t_share=t_share)
                        llm_last_call_at[sym] = latest_bar.ts

                        data_bus.add_decision(sym, decision)
                        
                        # 保存信号到 SQLite
                        store.save_signal(
                            symbol=sym,
                            ts=latest_bar.ts,
                            signal_type=event.signal.value,
                            price=latest_bar.close,
                            rule_reason=event.reason,
                            llm_action=decision.action,
                            llm_confidence=decision.confidence,
                            llm_reasons="; ".join(decision.reasons),
                            t_share=float(t_share_by_symbol.get(sym, 0.0))
                        )

                        cn_name_for_log = get_cn_name(sym)
                        name_tag = f"{cn_name_for_log} " if cn_name_for_log else ""
                        log.info(
                            "LLM decision for %s%s: action=%s state=%s conf=%.2f reasons=%s risks=%s plan=%s next=%s",
                            name_tag,
                            sym,
                            decision.action,
                            decision.position_state,
                            decision.confidence,
                            "; ".join(decision.reasons[:3]),
                            "; ".join(decision.risks[:3]),
                            "; ".join(
                                [
                                    f"target={decision.operation_plan.get('target_price')} stop={decision.operation_plan.get('stop_price')} share={decision.operation_plan.get('suggested_share')} window={decision.operation_plan.get('time_window')}"
                                ]
                            ),
                            "; ".join(decision.next_decision_point[:3]),
                        )

                        in_trading_hours = _is_trading_time_cn(now)
                        data_age_s = (now - latest_bar.ts).total_seconds()
                        is_fresh = data_age_s <= float(policy.max_data_age_seconds)

                        exec_share = 1.0 if float(decision.confidence) >= 0.8 else 0.5

                        def _clamp(v: float, lo: float, hi: float) -> float:
                            return max(lo, min(hi, v))

                        cur_t = float(t_share_by_symbol.get(sym, 0.0))

                        if decision.action == "SELL_PART":
                            new_t = _clamp(cur_t - exec_share, -1.0, 1.0)
                            t_share_by_symbol[sym] = new_t
                            # 卖出后清空买入价记录（或你可以选择保留，直到仓位清零）
                            if new_t >= 0:
                                last_buy_price_by_symbol[sym] = 0.0
                        elif decision.action == "BUY_BACK":
                            new_t = _clamp(cur_t + exec_share, -1.0, 1.0)
                            t_share_by_symbol[sym] = new_t
                            # 记录买入价用于盈亏计算（只在增加仓位时更新）
                            last_buy_price_by_symbol[sym] = float(latest_bar.close)
                        elif decision.action == "CANCEL_PLAN":
                            t_share_by_symbol[sym] = 0.0

                        cur_t2 = float(t_share_by_symbol.get(sym, 0.0))
                        if cur_t2 <= -0.999:
                            position_state_by_symbol[sym] = "HOLDING_CASH"
                        elif cur_t2 >= 0.999:
                            position_state_by_symbol[sym] = "HOLDING_STOCK"
                        else:
                            position_state_by_symbol[sym] = "HOLDING_STOCK"

                        action_side = "BUY" if decision.action == "BUY_BACK" else ("SELL" if decision.action == "SELL_PART" else None)

                        score = None
                        score_breakdown = None
                        score_features = None
                        accepted = None
                        replaced = None
                        should_emit = True
                        skip_reason = None

                        if policy.trading_hours_only and not in_trading_hours:
                            should_emit = False
                            skip_reason = "outside_trading_hours"

                        if should_emit and not is_fresh:
                            should_emit = False
                            skip_reason = "stale_data"

                        if action_side in {"BUY", "SELL"}:
                            score, score_breakdown, score_features = score_low_high(
                                side=action_side,
                                closes=closes,
                                snap=snap,
                                lookback=policy.lookback_bars,
                                near_extreme_ratio=policy.near_extreme_ratio,
                                volume_multiplier_buy=policy.volume_multiplier_buy,
                                hist_series=None,
                            )
                            w = policy.score_weights
                            final_score = (
                                float(score_breakdown.price_position) * float(w.get("price_position", 0.0))
                                + float(score_breakdown.macd_momentum) * float(w.get("macd_momentum", 0.0))
                                + float(score_breakdown.volume) * float(w.get("volume", 0.0))
                                + float(score_breakdown.ma_trend) * float(w.get("ma_trend", 0.0))
                            )
                            score = max(0.0, min(1.0, float(final_score)))

                            candidate = {
                                "symbol": sym,
                                "bar_ts": latest_bar.ts.isoformat(timespec="seconds"),
                                "price": float(latest_bar.close),
                                "action": action_side,
                                "score": float(score),
                                "llm_conf": float(decision.confidence),
                            }

                            accepted, replaced = daily_budget.consider(action_side, candidate)
                            if not accepted:
                                should_emit = False
                                skip_reason = "budget_not_selected"

                        cn_name = get_cn_name(sym)
                        decision_logger.append(
                            {
                                "ts_now": now.isoformat(timespec="seconds"),
                                "bar_ts": latest_bar.ts.isoformat(timespec="seconds"),
                                "symbol": sym,
                                "name": cn_name,
                                "data_source": source_tag,
                                "in_trading_hours": in_trading_hours,
                                "data_age_seconds": data_age_s,
                                "is_fresh": is_fresh,
                                "rule_event": {"signal": event.signal.value, "reason": event.reason},
                                "llm": {
                                    "action": decision.action,
                                    "confidence": float(decision.confidence),
                                    "reasons": decision.reasons,
                                    "risks": decision.risks,
                                    "position_state": decision.position_state,
                                    "t_share": float(t_share_by_symbol.get(sym, 0.0)),
                                    "total_share": 1.0 + float(t_share_by_symbol.get(sym, 0.0)),
                                    "operation_plan": decision.operation_plan,
                                    "next_decision_point": decision.next_decision_point,
                                },
                                "policy": {
                                    "trading_hours_only": policy.trading_hours_only,
                                    "max_data_age_seconds": policy.max_data_age_seconds,
                                    "lookback_bars": policy.lookback_bars,
                                    "budget": daily_budget.snapshot(),
                                },
                                "score": {
                                    "action_side": action_side,
                                    "value": score,
                                    "breakdown": (
                                        {
                                            "price_position": score_breakdown.price_position,
                                            "macd_momentum": score_breakdown.macd_momentum,
                                            "volume": score_breakdown.volume,
                                            "ma_trend": score_breakdown.ma_trend,
                                        }
                                        if score_breakdown is not None
                                        else None
                                    ),
                                    "features": score_features,
                                },
                                "budget_decision": {
                                    "accepted": accepted,
                                    "replaced": replaced,
                                },
                                "emit": {
                                    "should_emit": should_emit,
                                    "skip_reason": skip_reason,
                                },
                            }
                        )

                        if not should_emit:
                            log.info(
                                "alert skipped %s %s action=%s conf=%.2f score=%s reason=%s",
                                sym,
                                latest_bar.ts.strftime("%Y-%m-%d %H:%M"),
                                decision.action,
                                decision.confidence,
                                (f"{score:.3f}" if isinstance(score, float) else "NA"),
                                skip_reason,
                            )
                            continue

                        cn_name_for_reason = cn_name or ""
                        name_part = f"{cn_name_for_reason} " if cn_name_for_reason else ""
                        reason_prefix = f"{name_part}LLM action={decision.action} conf={decision.confidence:.2f}"
                        if decision.confidence < llm_min_conf:
                            reason_prefix += " (low_conf)"
                        if action_side in {"BUY", "SELL"} and isinstance(score, float):
                            reason_prefix += f" score={score:.3f}"

                        signal_type = SignalType.INFO
                        if decision.action == "BUY_BACK":
                            signal_type = SignalType.BUY_BACK
                        elif decision.action == "SELL_PART":
                            signal_type = SignalType.SELL_PART
                        elif decision.action == "CANCEL_PLAN":
                            signal_type = SignalType.CANCEL_PLAN
                        elif decision.action == "HOLD_POSITION":
                            signal_type = SignalType.HOLD_POSITION

                        position_info = f"[总仓={1.0 + float(t_share_by_symbol.get(sym, 0.0)):.1f} t_share={float(t_share_by_symbol.get(sym, 0.0)):+.1f}]"
                        full_reason = f"{reason_prefix} {position_info}; " + "; ".join(decision.reasons[:4])

                        emit(
                            event.__class__(
                                symbol=sym,
                                ts=latest_bar.ts,
                                signal=signal_type,
                                reason=full_reason,
                                dif=snap.dif,
                                dea=snap.dea,
                                hist=snap.hist,
                                vol=snap.vol,
                                vma=snap.vma,
                                close=latest_bar.close,
                            )
                        )
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
