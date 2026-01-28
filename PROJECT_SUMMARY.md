# 工程总结（股票分钟级监控 + LangChain 告警 Agent + thsdk 网关）

## 目标

- A 股分钟级（`1m`）监控与告警：基于 **MACD / 金叉死叉 / 成交量**等指标给出“做T”买卖提示。
- 当前实现为 **只告警**（不下单），并支持 **LLM 智能体（SiliconFlow / OpenAI 兼容）**对候选信号进行二次诊断与解释。
- 为解决云端/免费源不稳定问题，引入 **本地网关（FastAPI + thsdk）**作为优先行情源，`akshare` 作为兜底。

## 总体架构

数据流：

1. **数据源（primary: gateway, fallback: akshare）**
2. **BarStore 滑窗缓存**（保存最近 N 根分钟K）
3. **指标计算**（MACD / MA60 / VMA20）
4. **规则触发器**（金叉死叉确认、防抖、冷却、趋势过滤、量能过滤）
5. **LLM 决策层（LangChain）**：输出详细诊断（reasons / risks / plan）
6. **告警输出**：
   - 可读摘要（控制台）
   - JSONL（一行 JSON，便于日志采集）

## 关键模块说明

### 1) `src/gateway_server.py`（本地行情网关）

- FastAPI 服务，提供：
  - `GET /health`：健康检查，包含 `gateway_token_set` 状态
  - `GET /bars`：返回分钟K（JSON）
- 鉴权：
  - 使用 `Authorization: Bearer <GATEWAY_TOKEN>`
- 环境变量加载：
  - 启动时通过 `load_env()` 自动读取本地配置文件（见 `dotenv_loader.py`）
- thsdk 数据字段映射：
  - 兼容 `thsdk` DataFrame 列名：`开盘价/收盘价/最高价/最低价/成交量/总金额`（以及 `时间`）
- 股票代码：
  - 接受 `sz300058/sh600519` 或 `USZA300058/USHA600519`，内部会统一为 thsdk 代码

### 2) `src/agent_langchain.py`（LangChain 告警 Agent，定时盯盘）

- 定时循环读取 `config.yaml`：
  - 获取行情（`fetch_bars`）
  - 更新滑窗（`BarStore`）
  - 计算指标（`latest_snapshot`）
  - 规则触发（`SignalEngine.evaluate`）
  - 调用 LLM（LangChain + SiliconFlow）生成结构化 JSON 决策
- 输出：
  - 心跳（HEARTBEAT）：每轮每标的输出当前 close/指标/来源
  - 告警（ALERT）：可读摘要 + JSONL
- 测试模式：
  - `agent.test_mode=true`：忽略交易时段限制（便于非交易时段联调）
  - `agent.test_force_llm=true`：即使无规则信号也强制调用 LLM（验证全链路）

### 3) `src/datafeed.py`（数据源选择器）

- 按 `config.yaml` 选择主备数据源：
  - `primary: gateway`
  - `fallback: akshare`
- 返回 `(bars, source_tag)`，便于在心跳日志中展示实际数据来源。

### 4) `src/datafeed_gateway.py`

- 通过 HTTP 请求网关 `/bars` 获取分钟K
- 解析 JSON → 转 `Bar` 列表

### 5) `src/datafeed_akshare.py`

- akshare 作为兜底数据源：
  - 多 provider fallback（`hist_min_em` → `pre_min_em`）
  - 重试 + 熔断（circuit breaker）降低刷屏

### 6) `src/dotenv_loader.py`

- 由于某些环境对 `.env` dotfile 有限制，支持从以下路径加载：
  - `./.env`
  - `./src/.env`
  - `./src/env.local`（推荐）

### 7) 其他核心模块

- `src/bar_store.py`：分钟K滑窗缓存
- `src/indicators.py`：MACD/均线/均量等计算
- `src/signals.py`：规则引擎（金叉死叉确认/量能过滤/冷却/趋势开关）
- `src/ths_symbols.py`：`sz/sh` → `USZA/USHA` 代码转换
- `src/main.py`：旧版脚本入口（已保留），当前主要使用 `agent_langchain.py`

## 配置文件

### 1) `config.yaml`

- `symbols`：监控标的列表（如 `sz300058`）
- `poll_interval_seconds`：轮询频率
- `data_source`：数据源配置（网关优先 + akshare 兜底）
- `agent`：测试/心跳配置
- `llm_agent`：节流参数等

### 2) `src/env.local`（本地私密配置，勿提交）

用于本地加载环境变量（LLM、网关 token、可选 thsdk 登录）。示例字段：

- `LLM_API_BASE` / `LLM_API_KEY` / `LLM_MODEL`
- `GATEWAY_TOKEN`
- （可选）`THS_USERNAME/THS_PASSWORD/THS_MAC`

> 注意：密钥不要提交到仓库。游客账号模式下可不填 THS_*，thsdk 会使用临时账户（可能不稳定）。

## 运行方式

### 1) 启动网关（推荐先启动）

```bash
source .venv/bin/activate
python -m uvicorn src.gateway_server:app --host 0.0.0.0 --port 8000
```

验证：

```bash
curl -s http://127.0.0.1:8000/health | cat
export GATEWAY_TOKEN=$(python -c 'from src.dotenv_loader import load_env; import os; load_env(); print(os.getenv("GATEWAY_TOKEN") or "")')
curl -s -H "Authorization: Bearer $GATEWAY_TOKEN" "http://127.0.0.1:8000/bars?symbol=sz300058&limit=2&interval=1m" | cat
```

### 2) 启动 LangChain Agent（定时盯盘）

```bash
source .venv/bin/activate
python3 -m src.agent_langchain
```

非交易时段联调：
- 在 `config.yaml` 设置：
  - `agent.test_mode: true`
  - `agent.test_force_llm: true`

## 调试（VS Code launch）

已提供：`.vscode/launch.json`

- `Gateway (FastAPI) - uvicorn`
- `Agent (LangChain) - Test Mode`
- `Agent (LangChain) - Trading Hours`

## 已知注意事项

1. **Python 3.14 警告**：LangChain 在 3.14+ 会出现 pydantic v1 兼容性 warning。当前可跑，但建议长期使用 Python 3.12/3.13。
2. **thsdk 游客账号不稳定**：可能触发风控/验证码/失效。
3. **akshare 免费源不稳定**：作为兜底，不建议作为唯一实时源。
4. **密钥安全**：`LLM_API_KEY`、`GATEWAY_TOKEN`、THS 账号等不要提交到仓库。

