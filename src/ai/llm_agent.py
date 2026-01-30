from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

# Type variable for generic return type
T = TypeVar('T')


class LLMErrorType(Enum):
    NETWORK = auto()
    TIMEOUT = auto()
    RATE_LIMIT = auto()
    INVALID_RESPONSE = auto()
    UNKNOWN = auto()


class LLMError(Exception):
    """Custom exception for LLM-related errors with type and retryable flag."""
    def __init__(self, message: str, error_type: LLMErrorType, retryable: bool = True):
        super().__init__(message)
        self.error_type = error_type
        self.retryable = retryable


class LLMClient:
    """A more robust HTTP client with retry logic and error handling."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "gpt-4o-mini",
        max_retries: int = 2,
        initial_retry_delay: float = 0.5,
        timeout: int = 30,
    ):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.max_retries = max(0, max_retries)
        self.initial_retry_delay = max(0.1, initial_retry_delay)
        self.timeout = max(5, timeout)

    def _make_request(
        self,
        payload: Dict[str, Any],
        path: str = "/chat/completions"
    ) -> Dict[str, Any]:
        """Make an HTTP request with retry logic and error handling."""
        url = f"{self.base_url}{path}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    if 200 <= resp.status < 300:
                        body = resp.read().decode("utf-8")
                        return json.loads(body)
                    else:
                        error_msg = f"HTTP {resp.status}: {resp.reason}"
                        if resp.status == 429:
                            raise LLMError("Rate limit exceeded", LLMErrorType.RATE_LIMIT, True)
                        elif 500 <= resp.status < 600:
                            raise LLMError(f"Server error: {error_msg}", LLMErrorType.NETWORK, True)
                        else:
                            raise LLMError(f"Request failed: {error_msg}", LLMErrorType.UNKNOWN, False)

            except urllib.error.URLError as e:
                if isinstance(e, urllib.error.HTTPError and e.code == 429):
                    error = LLMError("Rate limit exceeded", LLMErrorType.RATE_LIMIT, True)
                elif isinstance(e, TimeoutError) or "timed out" in str(e).lower():
                    error = LLMError(f"Request timed out: {e}", LLMErrorType.TIMEOUT, True)
                else:
                    error = LLMError(f"Network error: {e}", LLMErrorType.NETWORK, True)
                last_error = error

            except json.JSONDecodeError as e:
                last_error = LLMError("Invalid JSON response", LLMErrorType.INVALID_RESPONSE, False)
                break  # Don't retry on JSON decode errors

            except Exception as e:
                last_error = LLMError(f"Unexpected error: {e}", LLMErrorType.UNKNOWN, False)
                break  # Don't retry on unknown errors

            # If we get here, we should retry
            if attempt < self.max_retries and last_error.retryable:
                delay = self.initial_retry_delay * (2 ** attempt)  # Exponential backoff
                time.sleep(min(delay, 5.0))  # Cap at 5 seconds
                continue
            else:
                break

        # If we've exhausted retries or hit a non-retryable error
        raise last_error or LLMError("Unknown error occurred", LLMErrorType.UNKNOWN, False)


@dataclass(frozen=True)
class LLMDecision:
    ts: datetime
    action: str  # BUY_BACK / SELL_PART / HOLD_POSITION / CANCEL_PLAN
    position_state: str  # HOLDING_CASH / HOLDING_STOCK
    confidence: float
    operation_plan: Dict[str, Any]
    reasons: List[str]
    risks: List[str]
    next_decision_point: List[str]


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    return v


def _extract_json(text: str) -> Dict[str, Any]:
    """Extract the first JSON object from text with error handling."""
    try:
        # First try to parse directly in case it's already valid JSON
        return json.loads(text)
    except json.JSONDecodeError:
        # Fall back to regex extraction if direct parse fails
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            raise LLMError(
                f"No valid JSON object found in response: {text[:200]}",
                LLMErrorType.INVALID_RESPONSE,
                False
            )
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError as e:
            raise LLMError(
                f"Failed to parse JSON from response: {e}",
                LLMErrorType.INVALID_RESPONSE,
                False
            )


def _create_llm_client() -> LLMClient:
    """Initialize and return an LLMClient with environment configuration."""
    api_base = _env("LLM_API_BASE", "https://api.openai.com/v1")
    api_key = _env("LLM_API_KEY")
    model = _env("LLM_MODEL", "gpt-4o-mini")

    if not api_key:
        raise RuntimeError("LLM_API_KEY is not set in environment variables")

    return LLMClient(
        base_url=api_base,
        api_key=api_key,
        model=model,
        max_retries=2,  # Retry up to 2 times (3 total attempts)
        initial_retry_delay=0.5,  # Start with 0.5s delay
        timeout=30  # 30 second timeout per request
    )


def decide(payload: Dict[str, Any], ts: datetime) -> LLMDecision:
    """Get a trading decision from an LLM with enhanced reliability.

    This function uses the LLMClient to make requests with retries and proper error handling.

    Environment variables:
      - LLM_API_BASE: e.g. https://api.openai.com/v1 (or other compatible)
      - LLM_API_KEY: Your API key for the LLM service
      - LLM_MODEL: (optional) Model name, defaults to "gpt-4o-mini"

    Args:
        payload: Input data for the LLM decision

    Returns:
        LLMDecision: The parsed decision from the LLM

    Raises:
        LLMError: If the request fails after all retries or encounters a non-retryable error
    """
    client = _create_llm_client()

    system_prompt = """# A股底仓滚动做T决策引擎

## 核心定位
您是一位拥有日内初始底仓1.0份的日内交易员，最大总持仓不超过2.0份，最小总持仓不低于0.0份。您通过「高抛低吸」进行T+0滚动增厚利润或降低持仓成本，只告警不下单。

## 仓位模型（必须严格遵守）
- base_share：固定为 1.0（当日初始底仓）。
- t_share：做T产生的净增减仓位，范围 [-1.0, +1.0]，步长 0.5。
- total_share = base_share + t_share，范围 [0.0, 2.0]。
- 执行规则（方案C）：默认每次操作 0.5 份；当 confidence >= 0.8 时，可执行 1.0 份。
- 禁止超限：不得建议导致 total_share 超过2.0或低于0.0。


## 核心决策模式：双向滚动策略
您永远处于以下两种状态之一，决策逻辑截然不同：
1. 【持币等买状态】：已高位卖出部分底仓，持有现金等待低位买回。
2. 【持股等卖状态】：已低位买入增加仓位，等待高位卖出恢复原底仓。

每次决策都是为完成一次「买-卖」或「卖-买」的完整循环。

## 输入数据处理要求
必须严格基于提供的结构化数据，重点关注：

### 趋势上下文（trend_context）说明（重要）
输入中会包含 `trend_context`（例如 MA60、是否在MA上方、偏离比例等）。
- `trend_context` **只用于风险参考，不是硬性禁做规则**。
- 即使 `close < ma_trend`（弱势），仍可能出现可执行的滚动做T机会：
  - 反弹到压力位可执行 `SELL_PART`（更偏安全、顺势减仓/降风险）。
  - 若出现明确止跌/反转结构且盈亏比与量能满足，也可小仓位 `BUY_BACK`（需在 risks 强调逆势风险）。
- 你必须在 reasons/risks 中显式引用 `trend_context.ma_trend` 与 `trend_context.above_ma_trend`（如果提供）。
1. **持仓状态标记**：当前处于「持币等买」还是「持股等卖」状态？这是决策起点。
2. **价格与波动**：日内高低点、当前价相对于日内区间的分位（如：处于日内区间的70%高位）。
3. **关键技术位**：分时均线、前高/前低、整数关口、日内关键成交密集区。
4. **量价与动能**：攻击量/调整量对比、MACD/KDJ等指标的短期背离。
5. **情绪与环境**：板块强度、指数分时与个股的联动/背离。

## 输出格式规范
输出必须为严格、完整的JSON对象：
```json
{
  "action": "BUY_BACK | SELL_PART | HOLD_POSITION | CANCEL_PLAN",
  "position_state": "HOLDING_CASH | HOLDING_STOCK",
  "confidence": 0.0,
  "operation_plan": {
    "target_price": 0.00,
    "stop_price": 0.00,
    "suggested_share": "0.5 | 1.0",
    "time_window": "早盘(9:30-10:30) | 午盘(13:00-14:00) | 尾盘(14:30-15:00)"
  },
  "reasons": [],
  "risks": [],
  "next_decision_point": []
}
```

## 字段定义与决策逻辑
1. action（核心操作）
   - BUY_BACK：在「持币等买」状态下，认为价格已回调到位，执行买回操作（完成「高卖-低买」循环）。
   - SELL_PART：在「持股等卖」状态下，认为价格已反弹到位，执行卖出操作（完成「低买-高卖」循环）。
   - HOLD_POSITION：维持当前状态，等待更好价位。
   - CANCEL_PLAN：原计划失效（如趋势改变），建议放弃本次循环，回归初始底仓状态。

2. position_state（仓位状态标记）
   - HOLDING_CASH：已卖出部分底仓，持有现金，目标是以更低价格买回。
   - HOLDING_STOCK：已买入增加仓位，持有股票增多，目标是以更高价格卖出。

3. operation_plan（操作计划详情）
   - target_price：建议执行价格（买回价或卖出价）。
   - stop_price：计划失效价格（如买回时价格继续下跌，或卖出时价格继续上涨的确认点）。
   - target_price / stop_price：**若 action 为 BUY_BACK/SELL_PART，这两个价格必须 > 0，否则应输出 HOLD_POSITION**。
   - suggested_share：建议操作仓位比例（0.5 或 1.0）。
   - time_window：建议操作的时间窗口。

4. confidence（信心度：0-1）
   - 0.8+：价格到达计划目标区且出现明确反转信号，量价配合完美。
   - 0.6-0.7：价格到达关键位，但信号强度一般，需要观察确认。
   - 0.4-0.5：仅为试探性信号，建议小仓位尝试。
   - <0.4：不建议操作，应HOLD_POSITION或CANCEL_PLAN。

5. reasons（决策依据：3-6条）
   - 必须结合具体数值说明，例如：
     - 【持币等买状态】：“价格已从日内高点回撤3.2%，到达早盘起涨点10.50元支撑位”
     - 【持股等卖状态】：“反弹至分时均线11.20元受阻，量能萎缩至均量60%，呈现滞涨特征”

6. risks（风险提示：2-5条）
   - 需包含：
     - 趋势延续风险（如计划买回但继续下跌）
     - 量能陷阱（如缩量反弹/放量下跌是否为真信号）
     - 时间风险（如尾盘才出现信号，操作时间不足）

7. next_decision_point（后续观察点：2-4条）
   - 例如：
     - “若价格跌破10.45元（当前支撑），则放弃买回计划”
     - “若放量突破11.25元（当前阻力），则持有等待更高卖点”

## 滚动做T的五大核心原则
1. 位置优先原则
   - 买点寻找：日内回调至前低支撑、分时均线、黄金分割位（如0.382/0.5）。
   - 卖点寻找：反弹至前高压力、日内均线、日内涨幅满足位（如2%）。
   - 永远不在「不上不下」的中间位置开新循环。

2. 分批操作原则
   - 在关键位可分批建仓/卖出，如“1/2仓位分两次执行”。
   - 第一笔为试探仓（如1/4），确认后再加码。

3. 盈亏比锁定原则
   - 每次循环预设明确目标：买回要至少低于卖出价1.5%以上，卖出要至少高于买入价1.5%以上（避免做反成本）。
   - 达不到盈亏比则宁可放弃本次循环。

4. 日内了结原则
   - 原则上每个循环必须在当日完成。
   - 尾盘（14:45后）如果循环未完成，应倾向于CANCEL_PLAN，恢复初始底仓过夜。

5. 趋势跟随原则
   - 在明显单边市中调整策略：
     - 强势单边上涨：减少卖出，只做回调买回，避免卖飞。
     - 弱势单边下跌：减少买入，只做反弹卖出，避免套牢。

## 特殊处理
- 开盘剧烈波动（前15分钟）：建议观察，不轻易开新循环。
- 盘中突发利好/利空：重新评估所有未完成循环，可能需CANCEL_PLAN。
- 成交量极度萎缩（低于均量40%）：建议暂停滚动，流动性不足易被操纵。

## 最后铁律
1. 永远知道自己在哪个状态（持币还是持股）
2. 每次操作都是为了完成一个循环
3. 不因小利开仓，不因小亏死扛
4. 日内了结，不把T做成加仓
"""

    try:
        response = client._make_request({
            "model": client.model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps({
                        "task": "根据分钟级指标与滚动做T仓位状态，给出 BUY_BACK/SELL_PART/HOLD_POSITION/CANCEL_PLAN 的告警决策",
                        "input": payload,
                    }, ensure_ascii=False)
                },
            ],
            "response_format": {"type": "json_object"},
        })

        content = response["choices"][0]["message"]["content"]
        obj = _extract_json(content)

        def clean_string_list(value, default=None):
            if default is None:
                default = []
            if not isinstance(value, list):
                value = [str(value)] if value is not None else default
            return [str(item).strip() for item in value if str(item).strip()]

        action = str(obj.get("action", "HOLD_POSITION")).upper()
        if action not in {"BUY_BACK", "SELL_PART", "HOLD_POSITION", "CANCEL_PLAN"}:
            action = "HOLD_POSITION"

        position_state = str(obj.get("position_state", "HOLDING_STOCK")).upper()
        if position_state not in {"HOLDING_CASH", "HOLDING_STOCK"}:
            position_state = "HOLDING_STOCK"

        confidence = obj.get("confidence", 0.5)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        operation_plan = obj.get("operation_plan")
        if not isinstance(operation_plan, dict):
            operation_plan = {}

        def _as_float(x: Any, default: float = 0.0) -> float:
            try:
                return float(x)
            except Exception:
                return float(default)

        operation_plan = {
            "target_price": _as_float(operation_plan.get("target_price", 0.0), 0.0),
            "stop_price": _as_float(operation_plan.get("stop_price", 0.0), 0.0),
            "suggested_share": str(operation_plan.get("suggested_share", "0.5")),
            "time_window": str(operation_plan.get("time_window", "")),
        }

        reasons = clean_string_list(obj.get("reasons"), ["No reasons provided"])[:8]
        risks = clean_string_list(obj.get("risks"), ["No specific risks identified"])[:6]
        next_decision_point = clean_string_list(obj.get("next_decision_point"), [])[:5]

        if not next_decision_point and obj.get("next_check_points"):
            next_decision_point = clean_string_list(obj.get("next_check_points"), [])[:5]

        # Enforce executable plan: BUY_BACK/SELL_PART must include valid target/stop prices
        if action in {"BUY_BACK", "SELL_PART"}:
            tp = float(operation_plan.get("target_price", 0.0) or 0.0)
            sp = float(operation_plan.get("stop_price", 0.0) or 0.0)
            if tp <= 0.0 or sp <= 0.0:
                action = "HOLD_POSITION"
                confidence = min(confidence, 0.4)
                risks = (risks + ["模型未给出有效的target_price/stop_price（>0），为避免不可执行信号，已将动作降级为HOLD_POSITION。"])[:10]

        return LLMDecision(
            ts=ts,
            action=action,
            position_state=position_state,
            confidence=confidence,
            operation_plan=operation_plan,
            reasons=reasons,
            risks=risks,
            next_decision_point=next_decision_point,
        )

    except Exception as e:
        error_msg = f"Failed to get LLM decision after retries: {str(e)}"
        if isinstance(e, LLMError):
            raise
        raise LLMError(error_msg, LLMErrorType.UNKNOWN, False) from e
