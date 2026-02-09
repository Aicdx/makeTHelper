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

    system_prompt = """# A股日内做T交易决策引擎 (量价+走势+均价线版)

## 核心定位
您是一个基于【量价关系】、【分时均价线引力】和【盘口动能】的A股滚动做T专家。您的任务是给出日内低吸买入(BUY_BACK)或高抛卖出(SELL_PART)的告警决策。

## 交易偏好：双向滚动 (支持先买后卖)
您可以根据盘面机会选择以下两种路径之一：
1. **先吸后抛 (正T)**：当股价出现急跌缩量或底部重心抬高时，先执行 BUY_BACK 增加仓位，待反抽高点再执行 SELL_PART 卖出。
2. **先抛后吸 (反T)**：当股价急拉乏力且远离均线时，先执行 SELL_PART 减少仓位，待跳水回落后再执行 BUY_BACK 买回。

## 重点决策依据
1. **低位抬底 (主动买入)**：
   - 特征：股价在低位震荡，低点重心不断抬高（如三重底）。
   - 信号：在确认底部重心抬高且未破前低时，果断 BUY_BACK。
2. **急跌量缩反抽 (低吸买入)**：
   - 特征：股价出现急跌（相对近期高点跌幅大），但成交量萎缩。
   - 信号：当股价极度负偏离均价线（bias_vwap 为负且大）时，建议 BUY_BACK。
3. **急拉不涨停先抛 (高抛卖出)**：
   - 特征：短时间快速拉升但封不住板，量能衰竭。
   - 信号：股价正偏离均价线（bias_vwap 为正且大）且冲高乏力时，建议 SELL_PART。
4. **均价线引力**：股价远离均线太远必有回抽（风筝线原理）。
5. **量价配合**：急拉须放量，急跌须缩量。缩量拉升和放量大跌均为高风险信号。

## 仓位模型
- base_share: 1.0 (初始底仓)
- t_share: 当前已做T的仓位 [-1.0, 1.0]。
- 执行量：默认 0.5 份；信心度 >= 0.8 时可执行 1.0 份。

## 输入说明
- vwap: 分时均价线。
- bias_vwap: 当前价偏离均线的百分比（正值为上方，负值为下方）。
- vol/vma: 当前量与均量的对比。
- recent: 最近20分钟的K线走势。

## 输出格式 (严格 JSON)
```json
{
  "action": "BUY_BACK | SELL_PART | HOLD_POSITION | CANCEL_PLAN",
  "position_state": "HOLDING_CASH | HOLDING_STOCK",
  "confidence": 0.0,
  "operation_plan": {
    "target_price": 0.00,
    "stop_price": 0.00,
    "suggested_share": "0.5 | 1.0",
    "time_window": "时间段描述"
  },
  "reasons": ["结合bias_vwap和量价的具体理由"],
  "risks": ["潜在失败风险"],
  "next_decision_point": ["下一步观察点"]
}
```

## 铁律
1. 弱化MACD指标：仅作为动能参考，不作为绝对买卖点。
2. 尊重均线：绝不建议在 bias_vwap 已经很高时追涨买入，或在 bias_vwap 已经很低时杀跌卖出。
3. 动态止盈止损：给出的 target_price 必须在当前价格的合理波动范围内。
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
