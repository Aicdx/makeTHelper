from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class LLMDecision:
    action: str  # BUY_T / SELL_T / HOLD / DISABLE_T
    confidence: float
    reasons: List[str]


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    return v


def _post_json(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def _extract_json(text: str) -> Dict[str, Any]:
    # Try to find the first JSON object in the response.
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError(f"No JSON object found in LLM response: {text[:2000]}")
    return json.loads(m.group(0))


def decide(payload: Dict[str, Any]) -> LLMDecision:
    """LLM decision via OpenAI-compatible chat.completions.

    Environment variables:
      - LLM_API_BASE: e.g. https://api.openai.com/v1 (or other compatible)
      - LLM_API_KEY
      - LLM_MODEL: e.g. gpt-4o-mini
    """

    api_base = _env("LLM_API_BASE", "https://api.openai.com/v1")
    api_key = _env("LLM_API_KEY")
    model = _env("LLM_MODEL", "gpt-4o-mini")

    if not api_key:
        raise RuntimeError("LLM_API_KEY is not set")

    system = (
        "你是一个A股日内做T的交易助手，但你只输出告警建议，不进行下单。"
        "你必须严格基于输入的结构化数据（最近分钟K线摘要、MACD、均线、成交量）进行判断。"
        "输出必须是JSON对象，字段: action, confidence, reasons。"
        "action只能是 BUY_T, SELL_T, HOLD, DISABLE_T 之一。"
        "confidence是0到1的小数。reasons是字符串数组，简洁列出3-8条理由。"
        "如果趋势不支持或数据不足，action=DISABLE_T 或 HOLD。"
    )

    user = {
        "task": "根据分钟级指标决定是否给出做T的买入/卖出告警",
        "input": payload,
    }

    url = f"{api_base.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    body = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        "response_format": {"type": "json_object"},
    }

    resp = _post_json(url, headers, body, timeout=30)
    content = resp["choices"][0]["message"]["content"]
    obj = _extract_json(content)

    action = str(obj.get("action", "HOLD")).upper()
    if action not in {"BUY_T", "SELL_T", "HOLD", "DISABLE_T"}:
        action = "HOLD"

    confidence = obj.get("confidence", 0.5)
    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    reasons = obj.get("reasons", [])
    if not isinstance(reasons, list):
        reasons = [str(reasons)]
    reasons = [str(r) for r in reasons][:10]

    return LLMDecision(action=action, confidence=confidence, reasons=reasons)

