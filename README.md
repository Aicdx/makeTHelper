# makeTHelper（A股底仓滚动做T告警系统）

本项目是一个面向 **A股分钟级数据** 的“底仓滚动做T”告警系统：

- 你每天有 **初始底仓 1.0 份**（base_share=1.0）
- 允许日内通过高抛低吸把总仓位在 **0.0 ~ 2.0 份**之间滚动
- 系统 **只告警，不下单**：输出建议动作、目标价/止损价、风险点、下一次观察点

---

## 1. 核心概念：Action（动作）与 State（状态/仓位）

### 1.1 仓位模型（最重要）
系统用 `base_share + t_share` 描述“底仓 + 做T净偏离”。

- **base_share**：固定为 `1.0`（当日初始底仓）
- **t_share**：做T造成的净增减仓位
  - 范围：`[-1.0, +1.0]`
  - 步长：`0.5`
- **total_share = base_share + t_share**
  - 范围：`[0.0, 2.0]`

理解示例：
- `t_share=0.0`  => 总仓位 `1.0`
- `t_share=+0.5` => 总仓位 `1.5`
- `t_share=+1.0` => 总仓位 `2.0`（上限）
- `t_share=-0.5` => 总仓位 `0.5`
- `t_share=-1.0` => 总仓位 `0.0`（下限）

这样自然支持你期望的滚动序列：
- `1 + 1 - 1`
- `1 - 1 + 1`
- `1 + 0.5 - 0.5 + 0.5 - 0.5 = 1`

### 1.2 Action（LLM 输出的核心动作）
LLM 输出字段 `action` 取值：

- **BUY_BACK**
  - 含义：在相对低位建议“买回/加仓”一份（实际执行份额由方案C决定）
  - 效果：总仓位增加（`t_share` 变大）

- **SELL_PART**
  - 含义：在相对高位建议“卖出一部分”
  - 效果：总仓位减少（`t_share` 变小）

- **HOLD_POSITION**
  - 含义：观望，不做操作

- **CANCEL_PLAN**
  - 含义：本轮计划失效/风险偏大，建议取消滚动计划并回到初始底仓结构
  - 效果：`t_share -> 0.0`（回归总仓位 1.0）

### 1.3 position_state（状态标记）
系统同时维护 `position_state`（主要用于给 LLM 作为上下文提示）：

- **HOLDING_STOCK**：偏“持股/可考虑卖出”的状态
- **HOLDING_CASH**：偏“持币/可考虑买回”的状态

注意：由于现在允许 `t_share` 在中间态（例如 0.0、±0.5），理论上你既可买也可卖。
项目中当前做法是：
- 在 `t_share` 达到极值附近时更明确：
  - `t_share <= -1.0` 侧更偏 `HOLDING_CASH`
  - `t_share >= +1.0` 侧更偏 `HOLDING_STOCK`
- 其它中间态默认保持 `HOLDING_STOCK`（避免过度切换造成混乱）

### 1.4 方案C执行规则（仓位步长由置信度决定）
系统执行份额采用“方案C”：

- **默认执行 0.5 份**
- 当 `confidence >= 0.8` 时，**允许执行 1.0 份**

> 代码层面会对 `t_share` 做 clamp，保证总仓位不会超过 `0.0~2.0`。

---

## 2. 项目整体买卖策略（从数据到告警的完整链路）

整体流程：

1. **数据拉取**（`src/data/datafeed.py` / gateway 或 akshare fallback）
2. **数据缓存**（`src/data/bar_store.py` 维护窗口）
3. **指标计算**（`src/indicators/` 计算 MACD、MA60、VMA 等）
4. **规则信号引擎**（`src/signals/signals.py`）
   - 目前主要基于 `dif/dea` 关系变化触发候选信号（买回/卖出）
   - 趋势过滤：当 `close < MA60` 时会产生 `CANCEL_PLAN` 候选（风险控制）
5. **LLM 决策**（`src/ai/llm_agent.py`）
   - 输入：指标快照 + 近期序列 + rule_event + trading_context（包含 position_state / t_share）
   - 输出：严格 JSON（action、confidence、operation_plan、reasons、risks、next_decision_point）
6. **告警策略过滤**（`src/utils/alert_policy.py`）
   - trading_hours_only / 数据新鲜度
   - `score_low_high` 低点/高点评分（仅对 BUY_BACK/SELL_PART 参与）
   - DailyBudget：每天 BUY/SELL 入选数量限制，支持替换
7. **emit 告警**（`src/utils/alerter.py`）
   - 最终输出 `SignalEvent`，并在 reason 中附带仓位信息
8. **日志**（JSONL）
   - `logs/trade_decisions_YYYY-MM-DD.jsonl` 追加写入，便于回测/复盘

---

## 3. LLM 输出 JSON 结构（关键字段说明）

`src/ai/llm_agent.py` 解析并产出 `LLMDecision`，主要字段：

- `action`: BUY_BACK | SELL_PART | HOLD_POSITION | CANCEL_PLAN
- `position_state`: HOLDING_CASH | HOLDING_STOCK
- `confidence`: 0~1
- `operation_plan`:
  - `target_price`: 建议执行价
  - `stop_price`: 计划失效/止损确认点
  - `suggested_share`: 0.5 或 1.0（提示；实际执行仍会按方案C以 confidence 决定）
  - `time_window`: 建议时间窗
- `reasons`: 3~6 条，必须引用具体数值/现象
- `risks`: 2~5 条，包含趋势/量能/时间等风险
- `next_decision_point`: 2~4 条，下一次验证条件

---

## 4. 如何运行

### 4.1 环境变量
需要在环境变量或 `.env` 中提供：

- `LLM_API_KEY`
- `LLM_API_BASE`（例如 OpenAI 兼容服务地址）
- `LLM_MODEL`（例如 `Qwen/Qwen2.5-72B-Instruct`）

如使用 gateway 拉取分钟数据，可能还需要：
- `GATEWAY_TOKEN`（对应 config.yaml 的 `token_env`）

### 4.2 启动主程序
在项目根目录运行：

```bash
python -m src.core.main
```

### 4.3 运行模式
由 `config.yaml` 控制：

- `agent.test_mode: true`：允许非交易时段也运行（便于测试）
- `agent.test_force_llm: true`：每只股票强制触发一次 LLM 决策（便于验证链路）

---

## 5. config.yaml 关键配置解释

### 5.1 symbols
需要监控的标的列表，例如：

```yaml
symbols:
  - sz300308
  - sh688270
```

### 5.2 poll_interval_seconds / window_size
- `poll_interval_seconds`: 轮询间隔秒数
- `window_size`: BarStore 保存的最大窗口大小

### 5.3 indicators
- `macd.fast / macd.slow / macd.signal`: MACD 参数
- `ma_trend`: 趋势均线长度（项目里通常作为 MA60）
- `vma`: 成交量均线窗口

### 5.4 strategy（SignalEngine）
- `confirm_bars`: 信号确认所需连续 bar 数
- `cooldown_minutes`: 同类信号冷却时间
- `volume_multiplier`: 放量阈值（用于过滤弱信号）
- `enable_trend_filter`: 是否开启趋势过滤（例如 close < MA60 时触发 CANCEL_PLAN）

### 5.5 llm_agent
- `enabled`: 是否启用 LLM
- `model`: 模型名（也可由环境变量 `LLM_MODEL` 覆盖）
- `min_confidence`: 低置信度时仍可记录但可能降低告警优先级
- `throttle_minutes`: 同一标的 LLM 调用的最小间隔（防止频繁调用）

### 5.6 alert_policy（告警策略）
- `trading_hours_only`: 只在交易时段发送
- `max_data_age_seconds`: 数据最大允许延迟
- `daily_budget`: 每日 BUY/SELL 入选上限以及替换规则
- `low_high`: 低点/高点评分相关参数
- `score_weights`: 各评分项权重
- `log.dir / log.file_prefix`: JSONL 日志落地目录与前缀

---

## 6. 风险提示
本项目用于交易告警研究与辅助决策，不构成任何投资建议。A股日内波动、滑点、数据延迟、执行偏差均可能导致策略失效，请自行承担风险。

