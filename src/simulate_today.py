import yaml
import pandas as pd
from datetime import datetime
import sys
import os

# 修复路径问题
sys.path.append(os.getcwd())

from src.data.datafeed_akshare import fetch_minute_bars
from src.indicators.types import Bar
from src.signals.signals import SignalEngine, SignalType
from src.indicators.indicators import latest_snapshot

def simulate_today():
    # 1. 加载配置
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    symbols = cfg.get("symbols", [])
    
    # 2. 初始化引擎
    strat_cfg = cfg.get("strategy", {})
    engine = SignalEngine(
        confirm_bars=int(strat_cfg.get("confirm_bars", 2)),
        cooldown_minutes=int(strat_cfg.get("cooldown_minutes", 15)),
        volume_multiplier=float(strat_cfg.get("volume_multiplier", 1.2)),
        enable_trend_filter=bool(strat_cfg.get("enable_trend_filter", True)),
    )

    ind_cfg = cfg.get("indicators", {})
    macd_cfg = ind_cfg.get("macd", {})
    ma_trend_n = int(ind_cfg.get("ma_trend", 60))
    vma_n = int(ind_cfg.get("vma", 20))

    print(f"=== 开始模拟今日行情策略测试 ({datetime.now().strftime('%Y-%m-%d')}) ===")

    for sym in symbols:
        print(f"\n正在抓取 {sym} 的今日数据...")
        # 抓取今天的分钟线 (akshare 模式)
        bars, status = fetch_minute_bars(sym)
        
        if not status.ok or not bars:
            print(f"  跳过: 无法获取 {sym} 的今日数据。原因: {status.reason}")
            continue
            
        print(f"  成功获取 {len(bars)} 条分钟线。开始模拟回放...")
        
        history_bars = []
        signals_found = 0
        
        today_open = bars[0].open if bars else None
        # 尝试从今日数据中获取昨收（如果包含昨日数据）或置为 None
        # 模拟脚本通常只抓取今日，这里暂时只提供 today_open
        
        for bar in bars:
            history_bars.append(bar)
            
            # 至少需要足够计算指标的数据量
            if len(history_bars) < max(30, vma_n + 5, ma_trend_n + 5):
                continue
                
            closes = [b.close for b in history_bars]
            vols = [b.volume for b in history_bars]
            
            snap = latest_snapshot(
                closes=closes,
                volumes=vols,
                macd_fast=int(macd_cfg.get("fast", 12)),
                macd_slow=int(macd_cfg.get("slow", 26)),
                macd_signal=int(macd_cfg.get("signal", 9)),
                ma_trend_n=ma_trend_n,
                vma_n=vma_n,
            )
            
            if snap is None:
                continue
                
            # 评估信号
            event = engine.evaluate(
                symbol=sym,
                ts=bar.ts,
                close=bar.close,
                ind=snap,
                prev_close=None, # simulate_today 暂不支持获取昨收，维持原逻辑或手动补充
                open_price=today_open
            )
            
            if event and event.signal != SignalType.INFO:
                signals_found += 1
                color = "\033[92m" if "BUY" in event.signal.value else "\033[91m"
                reset = "\033[0m"
                print(f"  {color}[{event.signal.value}]{reset} 时间: {bar.ts.strftime('%H:%M')} 价格: {bar.close:.3f} 原因: {event.reason}")

        if signals_found == 0:
            print("  该标的今日策略逻辑未触发任何买卖点。")

    print("\n=== 模拟测试完成 ===")

if __name__ == "__main__":
    simulate_today()

