from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List

import yaml
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.data.bar_store import BarStore
from src.data.datafeed import fetch_bars
from src.utils.dotenv_loader import load_env
from src.indicators.indicators import latest_snapshot
from src.signals.signals import SignalEngine
from src.utils.symbols import normalize_symbol
from src.indicators.types import Bar


class AgentDecision(BaseModel):
    action: str = Field(..., description="BUY_T/SELL_T/HOLD/DISABLE_T")
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasons: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    suggested_plan: List[str] = Field(default_factory=list, description="建议的操作步骤（只告警，不下单）")


@tool
def format_alert(symbol: str, decision_json: str, context_json: str) -> str:
    """Format alert output as a readable JSON line for logs."""
    try:
        decision = json.loads(decision_json)
    except Exception:
        decision = {"raw": decision_json}
    try:
        ctx = json.loads(context_json)
    except Exception:
        ctx = {"raw": context_json}

    out = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "symbol": symbol,
        "decision": decision,
        "context": ctx,
    }
    return json.dumps(out, ensure_ascii=False)


def _build_llm() -> ChatOpenAI:
    # Load env.local/.env if present (best effort)
    load_env()

    api_base = os.getenv("LLM_API_BASE", "https://api.siliconflow.cn/v1")
    api_key = os.getenv("LLM_API_KEY")
    model = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-72B-Instruct")
    if not api_key:
        raise RuntimeError("LLM_API_KEY is not set")
    return ChatOpenAI(model=model, api_key=api_key, base_url=api_base, temperature=0.2)


def _is_trading_time_cn(now: datetime) -> bool:
    hm = now.hour * 60 + now.minute
    morning = 9 * 60 + 30 <= hm <= 11 * 60 + 30
    afternoon = 13 * 60 <= hm <= 15 * 60
    return morning or afternoon


def _summarize_bars(bars: List[Bar], n: int = 20) -> Dict[str, Any]:
    tail = bars[-n:] if len(bars) > n else bars
    return {
        "n": len(bars),
        "tail": [
            {
                "ts": b.ts.strftime("%Y-%m-%d %H:%M"),
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
                "amount": b.amount,
            }
            for b in tail
        ],
    }


def _agent_prompt() -> str:
    return (
        "你是一个A股日内做T的交易告警智能体（只告警，不下单）。\n"
        "你会收到：分钟K线摘要、MACD/均线/量能指标、以及规则触发的候选信号（如金叉/死叉/趋势禁做T）。\n"
        "你的任务：输出一个严格JSON对象，符合给定schema（action/confidence/reasons/risks/suggested_plan）。\n"
        "要求：\n"
        "- action 只能是 BUY_T/SELL_T/HOLD/DISABLE_T\n"
        "- reasons 3~8条，必须引用输入中的具体数值/现象（例如 close 与 MA60、dif/dea/hist、量能对比等）\n"
        "- risks 2~6条，提醒可能的失败情形（假信号、缩量、趋势下行、数据延迟等）\n"
        "- suggested_plan 3~8条，给出‘只告警’的执行建议（例如等待确认/设定冷却/观察量能/分批做T等）\n"
        "- 若数据不足或源不稳定，优先输出 HOLD 或 DISABLE_T，并在 risks 中说明\n"
    )


def _print_heartbeat(sym: str, latest_bar: Bar, snap: Any, source_tag: str) -> None:
    ma = snap.ma_trend
    vma = snap.vma
    vol = snap.vol
    print(
        f"[HEARTBEAT {datetime.now():%H:%M:%S}] {sym} ts={latest_bar.ts:%Y-%m-%d %H:%M} "
        f"close={latest_bar.close:.3f} dif={snap.dif:.4f} dea={snap.dea:.4f} hist={snap.hist:.4f} "
        f"vol={(f'{vol:.0f}' if vol is not None else 'NA')} vma={(f'{vma:.0f}' if vma is not None else 'NA')} "
        f"ma={(f'{ma:.3f}' if ma is not None else 'NA')} source={source_tag}"
    )


def _print_alert_summary(sym: str, decision_obj: Dict[str, Any], ctx: Dict[str, Any]) -> None:
    action = decision_obj.get("action", "HOLD")
    conf = decision_obj.get("confidence", "?")
    reasons = decision_obj.get("reasons", [])
    risks = decision_obj.get("risks", [])
    plan = decision_obj.get("suggested_plan", [])

    print("=" * 80)
    print(f"[ALERT {datetime.now():%Y-%m-%d %H:%M:%S}] {sym} action={action} confidence={conf}")
    print(f"rule_event={ctx.get('rule_event')} data_source={ctx.get('data_source')}")

    if isinstance(reasons, list) and reasons:
        print("reasons:")
        for r in reasons[:8]:
            print(f"- {r}")

    if isinstance(risks, list) and risks:
        print("risks:")
        for r in risks[:6]:
            print(f"- {r}")

    if isinstance(plan, list) and plan:
        print("suggested_plan:")
        for r in plan[:8]:
            print(f"- {r}")

    print("=" * 80)


def run_watch(cfg_path: str = "config.yaml") -> None:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    symbols = [normalize_symbol(s) for s in cfg.get("symbols", [])]
    poll_interval = int(cfg.get("poll_interval_seconds", 15))
    window_size = int(cfg.get("window_size", 600))

    agent_cfg = cfg.get("agent", {})
    heartbeat_every_loops = int(agent_cfg.get("heartbeat_every_loops", 1))
    test_mode = bool(agent_cfg.get("test_mode", False))
    test_force_llm = bool(agent_cfg.get("test_force_llm", False))

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

    llm = _build_llm()

    store = BarStore(window_size=window_size)

    llm_throttle_min = int(cfg.get("llm_agent", {}).get("throttle_minutes", 5))
    llm_last_call: Dict[str, datetime] = {}

    loop_i = 0
    while True:
        now = datetime.now()
        if not test_mode and not _is_trading_time_cn(now):
            time.sleep(min(poll_interval, 30))
            continue

        loop_i += 1
        for sym in symbols:
            bars, source_tag = fetch_bars(cfg, sym)
            if not bars:
                if loop_i % heartbeat_every_loops == 0:
                    print(f"[HEARTBEAT {datetime.now():%H:%M:%S}] {sym} no bars source={source_tag}")
                continue

            store.upsert_bars(sym, bars)
            win = store.get_window(sym, 300)
            if len(win) < max(30, vma_n + 5, ma_trend_n + 5):
                if loop_i % heartbeat_every_loops == 0:
                    print(f"[HEARTBEAT {datetime.now():%H:%M:%S}] {sym} insufficient window={len(win)}")
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
                continue

            latest_bar = win[-1]
            if loop_i % heartbeat_every_loops == 0:
                _print_heartbeat(sym, latest_bar, snap, source_tag)

            event = engine.evaluate(sym, latest_bar.ts, latest_bar.close, snap)
            if event is None and not test_force_llm:
                continue

            last = llm_last_call.get(sym)
            if last is not None and (latest_bar.ts - last).total_seconds() < llm_throttle_min * 60:
                continue

            ctx = {
                "symbol": sym,
                "ts": latest_bar.ts.strftime("%Y-%m-%d %H:%M"),
                "close": latest_bar.close,
                "data_source": source_tag,
                "rule_event": {"signal": event.signal.value, "reason": event.reason} if event else {"signal": "TEST", "reason": "test_force_llm"},
                "indicators": asdict(snap),
                "bars": _summarize_bars(win, n=20),
            }

            schema = AgentDecision.model_json_schema()
            prompt = (
                _agent_prompt()
                + "\n\n输入JSON：\n"
                + json.dumps(ctx, ensure_ascii=False)
                + "\n\n输出schema：\n"
                + json.dumps(schema, ensure_ascii=False)
            )

            resp = llm.invoke(prompt)
            text = resp.content if hasattr(resp, "content") else str(resp)

            try:
                decision_obj = json.loads(text)
            except Exception:
                decision_obj = {"raw": text}

            _print_alert_summary(sym, decision_obj, ctx)

            line = format_alert.invoke(
                {
                    "symbol": sym,
                    "decision_json": json.dumps(decision_obj, ensure_ascii=False),
                    "context_json": json.dumps(ctx, ensure_ascii=False),
                }
            )
            print(line)

            llm_last_call[sym] = latest_bar.ts

        time.sleep(poll_interval)


if __name__ == "__main__":
    run_watch()
