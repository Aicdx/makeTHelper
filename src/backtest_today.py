import yaml
import sqlite3
import pandas as pd
from datetime import datetime, time
import sys
import os

# 修复路径问题，确保能找到 src 模块
sys.path.append(os.getcwd())

from src.indicators.types import Bar
from src.signals.signals import SignalEngine, SignalType
from src.indicators.indicators import latest_snapshot
from src.signals.decision import DecisionContext, llm_decision
from src.utils.dotenv_loader import load_env

def is_trading_hours(ts: datetime) -> bool:
    """判断是否在 A 股交易时段"""
    t = ts.time()
    morning = time(9, 30) <= t <= time(11, 30)
    afternoon = time(13, 0) <= t <= time(15, 0)
    return morning or afternoon

def run_backtest_today_with_llm():
    # 1. 加载环境与配置
    load_env()
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    llm_model = os.getenv("LLM_MODEL", cfg.get("llm_agent", {}).get("model", "unknown"))
    symbols = cfg.get("symbols", [])
    db_path = "data/bars.db"
    
    if not os.path.exists(db_path):
        print(f"错误: 数据库文件 {db_path} 不存在。")
        return

    # 模拟仓位管理
    position_states = {sym: "HOLDING_STOCK" for sym in symbols}
    t_shares = {sym: 0.0 for sym in symbols}

    # 2. 初始化策略引擎 (放宽测试门槛：即刻触发，不强制放量，不启用趋势过滤)
    engine = SignalEngine(
        confirm_bars=1,              # 1根确认，即刻进入LLM
        cooldown_minutes=5,          # 缩短冷却时间
        volume_multiplier=0.8,       # 极低量能要求
        enable_trend_filter=False    # 禁用趋势过滤，让LLM全权负责
    )

    ind_cfg = cfg.get("indicators", {})
    macd_cfg = ind_cfg.get("macd", {})
    ma_trend_n = int(ind_cfg.get("ma_trend", 60))
    vma_n = int(ind_cfg.get("vma", 20))

    print(f"=== 开始 [规则触发 + LLM] 今日回测 ({datetime.now().strftime('%Y-%m-%d')}) ===")
    print(f"当前使用的模型: \033[96m{llm_model}\033[0m")
    print(f"提示: 只有满足规则确认 (confirm_bars={engine.confirm_bars}) 且不在冷却期时才会调用 LLM。\n")

    conn = sqlite3.connect(db_path)
    today_str = datetime.now().strftime("%Y-%m-%d")

    for sym in symbols:
        print(f"\n>>> 分析标的: {sym}")
        # 获取今天的数据
        query = f"SELECT * FROM bars WHERE symbol = '{sym}' AND ts LIKE '{today_str}%' ORDER BY ts ASC"
        df = pd.read_sql_query(query, conn)
        
        if df.empty:
            print(f"  跳过: 数据库中没有 {sym} 今天的历史数据。")
            continue
            
        print(f"  找到 {len(df)} 条今日 K 线数据。")
        
        history_bars = []
        rule_hits = 0
        
        for i, row in df.iterrows():
            ts = datetime.fromisoformat(row['ts'])
            bar = Bar(ts=ts, open=row['open'], high=row['high'], low=row['low'], close=row['close'], volume=row['volume'], amount=row['amount'])
            history_bars.append(bar)
            
            # 过滤非交易时间
            if not is_trading_hours(ts):
                continue

            if len(history_bars) < max(30, vma_n + 5, ma_trend_n + 5):
                continue
                
            closes = [b.close for b in history_bars]
            vols = [b.volume for b in history_bars]
            amounts = [getattr(b, "amount", b.close * b.volume) for b in history_bars]
            
            snap = latest_snapshot(
                closes=closes, volumes=vols, amounts=amounts,
                macd_fast=int(macd_cfg.get("fast", 12)), macd_slow=int(macd_cfg.get("slow", 26)),
                macd_signal=int(macd_cfg.get("signal", 9)), ma_trend_n=ma_trend_n, vma_n=vma_n
            )
            
            if snap is None: continue
                
            # 评估规则
            event = engine.evaluate(symbol=sym, ts=bar.ts, close=bar.close, ind=snap)
            
            # 监控：如果 MACD 刚交叉或放量，但还没被 engine 确认，我们打印一条调试信息
            # (这能帮你理解为什么有些点位没触发 LLM)
            if snap.hist * (closes[-2] - closes[-3]) > 0 and abs(snap.hist) < 0.05:
                 # 接近交叉点位的监控
                 pass 

            if event and event.signal != SignalType.INFO:
                rule_hits += 1
                print(f"\n  \033[93m[规则触发]\033[0m 时间: {bar.ts.strftime('%H:%M')} 价格: {bar.close:.3f} 信号: {event.signal.value}")
                print(f"  原因: {event.reason}")
                print(f"  指标: DIF={snap.dif:.3f} DEA={snap.dea:.3f} HIST={snap.hist:.3f} MA{ma_trend_n}={snap.ma_trend:.3f}")
                
                print(f"  \033[94m[LLM 决策中...]\033[0m")
                try:
                    ctx = DecisionContext(
                        symbol=sym, ts=bar.ts, close=bar.close, indicators=snap,
                        recent_closes=closes[-100:], recent_volumes=vols[-100:],
                        recent_timestamps=[b.ts.strftime("%H:%M") for b in history_bars[-100:]],
                        rule_event={"signal": event.signal.value, "reason": event.reason}
                    )
                    
                    decision = llm_decision(ctx, position_state=position_states[sym], t_share=t_shares[sym])
                    
                    color = "\033[92m" if decision.action == "BUY_BACK" else ("\033[91m" if decision.action == "SELL_PART" else "\033[90m")
                    reset = "\033[0m"
                    print(f"  {color}[LLM 决策: {decision.action}]{reset} (置信度: {decision.confidence:.2f})")
                    print(f"  - 核心理由: {'; '.join(decision.reasons[:3])}")
                    if decision.risks:
                        print(f"  - 风险提示: {'; '.join(decision.risks[:2])}")
                    
                    # 模拟更新仓位，影响后续决策
                    if decision.action == "SELL_PART":
                        t_shares[sym] = max(-1.0, t_shares[sym] - 0.5)
                        if t_shares[sym] <= -0.9: position_states[sym] = "HOLDING_CASH"
                    elif decision.action == "BUY_BACK":
                        t_shares[sym] = min(1.0, t_shares[sym] + 0.5)
                        if t_shares[sym] >= 0.9: position_states[sym] = "HOLDING_STOCK"
                        
                except Exception as e:
                    print(f"  \033[31m[LLM 错误]\033[0m: {e}")

        if rule_hits == 0:
            print(f"  结果: 该标的今日全天未满足规则触发条件。")

    conn.close()
    print("\n=== 回测完成 ===")

if __name__ == "__main__":
    run_backtest_today_with_llm()
