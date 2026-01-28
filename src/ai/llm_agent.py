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
    action: str  # BUY_T / SELL_T / HOLD / DISABLE_T
    confidence: float
    reasons: List[str]
    risks: List[str]
    suggested_plan: List[str]


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


def decide(payload: Dict[str, Any]) -> LLMDecision:
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
    
    system_prompt = (
        "你是一个A股日内做T的交易告警助手（只告警，不下单）。"
        "你必须严格基于输入的结构化数据（最近分钟K线摘要、MACD、均线、成交量、规则触发信号）进行判断。"
        "输出必须是JSON对象，字段: action, confidence, reasons, risks, suggested_plan。"
        "action只能是 BUY_T, SELL_T, HOLD, DISABLE_T 之一。"
        "confidence是0到1的小数。"
        "reasons是字符串数组，3-8条，必须尽量引用输入中的具体数值/现象。"
        "risks是字符串数组，2-6条，提醒失败情形（假信号、缩量、趋势下行、数据延迟等）。"
        "suggested_plan是字符串数组，3-8条，给出只告警的执行建议（等待确认/观察量能/分批做T等）。"
        "如果趋势不支持或数据不足，action=DISABLE_T 或 HOLD，并在risks中说明。"
    )
    
    try:
        # Make the request with retry logic handled by LLMClient
        response = client._make_request({
            "model": client.model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user", 
                    "content": json.dumps({
                        "task": "根据分钟级指标决定是否给出做T的买入/卖出告警",
                        "input": payload,
                    }, ensure_ascii=False)
                },
            ],
            "response_format": {"type": "json_object"},
        })
        
        # Extract and validate the response
        content = response["choices"][0]["message"]["content"]
        obj = _extract_json(content)
        
        # Parse and validate the action
        action = str(obj.get("action", "HOLD")).upper()
        if action not in {"BUY_T", "SELL_T", "HOLD", "DISABLE_T"}:
            action = "HOLD"
            
        # Parse and validate confidence
        confidence = obj.get("confidence", 0.5)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))
        
        # Parse and clean reasons, risks, and suggested_plan
        def clean_string_list(value, default=None):
            if default is None:
                default = []
            if not isinstance(value, list):
                value = [str(value)] if value is not None else default
            return [str(item).strip() for item in value if str(item).strip()]
            
        reasons = clean_string_list(obj.get("reasons"), ["No reasons provided"])[:10]
        risks = clean_string_list(obj.get("risks"), ["No specific risks identified"])[:10]
        suggested_plan = clean_string_list(obj.get("suggested_plan"), ["No specific plan suggested"])[:12]
        
        return LLMDecision(
            action=action,
            confidence=confidence,
            reasons=reasons,
            risks=risks,
            suggested_plan=suggested_plan,
        )
        
    except Exception as e:
        # If we get here, all retries have been exhausted or we hit a non-retryable error
        error_msg = f"Failed to get LLM decision after retries: {str(e)}"
        if isinstance(e, LLMError):
            raise  # Re-raise our custom errors as-is
        else:
            # Wrap unexpected errors in our custom error type
            raise LLMError(
                error_msg,
                LLMErrorType.UNKNOWN,
                False  # Don't retry on unknown errors
            ) from e

