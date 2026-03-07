"""
通用工具函数：日志、异步限流、重试、JSON 健壮解析。
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import time
from functools import wraps
from typing import Any, Callable


# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# ---------------------------------------------------------------------------
# 异步令牌桶限速器（Token Bucket Rate Limiter）
# ---------------------------------------------------------------------------

class AsyncRateLimiter:
    """
    协程安全的令牌桶限速器。
    以 requests_per_minute 为上限，均匀分发请求。
    """

    def __init__(self, requests_per_minute: int):
        self.rpm = requests_per_minute
        self._min_interval = 60.0 / max(requests_per_minute, 1)
        self._last_call: float = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._min_interval - (now - self._last_call)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call = time.monotonic()


# ---------------------------------------------------------------------------
# 异步重试装饰器（指数退避）
# ---------------------------------------------------------------------------

class NonRetryableAPIError(RuntimeError):
    """不可重试的 API 错误（如欠费、鉴权失败、模型不存在）。"""


class RetryExhaustedAPIError(RuntimeError):
    """达到最大重试次数后仍失败。"""


_NON_RETRYABLE_ERROR_HINTS = (
    "insufficient_quota",
    "quota exceeded",
    "quota has been exceeded",
    "billing",
    "payment required",
    "insufficient balance",
    "balance is not enough",
    "account suspended",
    "account deactivated",
    "invalid api key",
    "authentication",
    "unauthorized",
    "forbidden",
    "invalid model",
    "model not found",
    "does not exist",
    "no such model",
    "欠费",
    "余额不足",
    "配额不足",
)


def is_non_retryable_api_error(exc: Exception) -> bool:
    """基于错误文案判断是否属于不可重试错误。"""
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(hint in text for hint in _NON_RETRYABLE_ERROR_HINTS)


def async_retry(
    max_attempts: int = 3,
    wait_seconds: float = 5.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    should_retry: Callable[[Exception], bool] | None = None,
):
    """
    装饰异步函数，失败时按指数退避重试最多 max_attempts 次。
    """
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return await fn(*args, **kwargs)
                except exceptions as exc:
                    if should_retry is not None and not should_retry(exc):
                        logger = get_logger("retry")
                        logger.error(
                            f"[{fn.__name__}] 命中不可重试错误，立即终止: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        raise NonRetryableAPIError(str(exc)) from exc
                    last_exc = exc
                    delay = wait_seconds * (2 ** (attempt - 1))
                    logger = get_logger("retry")
                    logger.warning(
                        f"[{fn.__name__}] 第 {attempt}/{max_attempts} 次失败: "
                        f"{type(exc).__name__}: {exc}. {delay:.1f}s 后重试..."
                    )
                    if attempt < max_attempts:
                        await asyncio.sleep(delay)
            raise RetryExhaustedAPIError(
                f"[{fn.__name__}] 已重试 {max_attempts} 次，均告失败。"
            ) from last_exc
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# 健壮 JSON 解析（应对 LLM 输出噪声）
# ---------------------------------------------------------------------------

def extract_json_from_llm_response(text: str) -> dict[str, str]:
    """
    从 LLM 的原始输出中稳健地提取 JSON 对象。

    策略（按优先级）：
    1. 直接 json.loads
    2. 提取 ```json ... ``` 代码块
    3. 正则找第一个 { ... } 块
    4. 抛出 ValueError
    """
    # 1. 直接解析
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. 代码块
    code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_block:
        try:
            return json.loads(code_block.group(1))
        except json.JSONDecodeError:
            pass

    # 3. 首个 { ... } 对象（贪婪，允许嵌套）
    brace_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"无法从 LLM 响应中解析出 JSON。原文片段:\n{text[:500]}")


# ---------------------------------------------------------------------------
# 成本估算（仅估算，非精确）
# ---------------------------------------------------------------------------

AVG_CHARS_PER_TOKEN = 4.0   # 英文平均值

def estimate_token_count(text: str) -> int:
    """粗略估计 token 数（无需加载 tiktoken）。"""
    return max(1, int(len(text) / AVG_CHARS_PER_TOKEN))


# Cost per 1M tokens (input / output), USD
_COST_TABLE: dict[str, tuple[float, float]] = {
    "gpt-4o-mini":                  (0.15, 0.60),
    "gpt-4o":                       (2.50, 10.00),
    "claude-3-5-haiku-20241022":    (0.80, 4.00),
    "claude-3-5-sonnet-20241022":   (3.00, 15.00),
    "gemini-2.0-flash":             (0.10, 0.40),
    "MiniMax-M2.5":                 (0.10, 1.10),
    "MiniMax-M2.5-highspeed":       (0.10, 1.10),
}

# Qwen 分档价格（单位：CNY / 1M tokens），按单次请求输入 token 分档
# 汇率固定使用 1 USD = 7 CNY
_CNY_PER_USD = 7.0
_QWEN35_PLUS_TIERS_CNY: list[tuple[int, float, float]] = [
    # (max_input_tokens, input_price_cny_per_1m, output_price_cny_per_1m)
    (128_000, 0.8, 4.8),
    (256_000, 2.0, 12.0),
    (1_000_000, 4.0, 24.0),
]
_QWEN35_FLASH_TIERS_CNY: list[tuple[int, float, float]] = [
    # (max_input_tokens, input_price_cny_per_1m, output_price_cny_per_1m)
    (128_000, 0.2, 2.0),
    (256_000, 0.8, 8.0),
    (1_000_000, 1.2, 12.0),
]


def _qwen35_plus_rates_usd(input_tokens: int) -> tuple[float, float]:
    """按输入 token 所处分档返回 qwen3.5-plus 的 (输入单价, 输出单价), USD/1M。"""
    for max_inp, inp_cny, out_cny in _QWEN35_PLUS_TIERS_CNY:
        if input_tokens <= max_inp:
            return inp_cny / _CNY_PER_USD, out_cny / _CNY_PER_USD
    inp_cny, out_cny = _QWEN35_PLUS_TIERS_CNY[-1][1], _QWEN35_PLUS_TIERS_CNY[-1][2]
    return inp_cny / _CNY_PER_USD, out_cny / _CNY_PER_USD


def _qwen35_flash_rates_usd(input_tokens: int) -> tuple[float, float]:
    """按输入 token 所处分档返回 qwen3.5-flash 的 (输入单价, 输出单价), USD/1M。"""
    for max_inp, inp_cny, out_cny in _QWEN35_FLASH_TIERS_CNY:
        if input_tokens <= max_inp:
            return inp_cny / _CNY_PER_USD, out_cny / _CNY_PER_USD
    inp_cny, out_cny = _QWEN35_FLASH_TIERS_CNY[-1][1], _QWEN35_FLASH_TIERS_CNY[-1][2]
    return inp_cny / _CNY_PER_USD, out_cny / _CNY_PER_USD


_QWEN35_FLASH_MODEL_IDS = {"qwen3.5-flash", "qwen3.5-flash-2026-02-23"}


def estimate_request_cost_usd(
    model_id: str,
    input_tokens: int,
    output_tokens: int,
) -> float | None:
    """按单次请求 token 估算费用（USD），未知模型返回 None。"""
    if model_id == "qwen3.5-plus":
        inp_rate, out_rate = _qwen35_plus_rates_usd(input_tokens)
    elif model_id in _QWEN35_FLASH_MODEL_IDS:
        inp_rate, out_rate = _qwen35_flash_rates_usd(input_tokens)
    else:
        if model_id not in _COST_TABLE:
            return None
        inp_rate, out_rate = _COST_TABLE[model_id]
    return input_tokens / 1_000_000 * inp_rate + output_tokens / 1_000_000 * out_rate


def estimate_cost_usd(
    model_id: str,
    total_input_tokens: int,
    total_output_tokens: int,
) -> float:
    """返回预估费用（美元）。"""
    detail = estimate_cost_breakdown_usd(model_id, total_input_tokens, total_output_tokens)
    if detail is None:
        return -1.0
    return detail["total_cost_usd"]


def estimate_cost_breakdown_usd(
    model_id: str,
    total_input_tokens: int,
    total_output_tokens: int,
) -> dict[str, float] | None:
    """
    返回费用分解（输入/输出/总计），未知模型返回 None。

    注意：
    - qwen3.5-plus / qwen3.5-flash 为分档计费，
      该函数在聚合 token 上估算时会按“总输入 token 对应分档”近似。
    - 更精确的估算应逐请求调用 estimate_request_cost_usd 后累加。
    """
    if model_id == "qwen3.5-plus":
        inp_cost, out_cost = _qwen35_plus_rates_usd(total_input_tokens)
    elif model_id in _QWEN35_FLASH_MODEL_IDS:
        inp_cost, out_cost = _qwen35_flash_rates_usd(total_input_tokens)
    else:
        if model_id not in _COST_TABLE:
            return None
        inp_cost, out_cost = _COST_TABLE[model_id]

    input_cost_usd = total_input_tokens / 1_000_000 * inp_cost
    output_cost_usd = total_output_tokens / 1_000_000 * out_cost
    return {
        "input_rate_per_1m": inp_cost,
        "output_rate_per_1m": out_cost,
        "input_cost_usd": input_cost_usd,
        "output_cost_usd": output_cost_usd,
        "total_cost_usd": input_cost_usd + output_cost_usd,
    }
