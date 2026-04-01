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

_JSON_VALID_ESCAPES = frozenset('"\\\/bfnrtu')


def _fix_invalid_json_escapes(text: str) -> str:
    """将 JSON 字符串中的非法反斜杠转义（如 LaTeX 的 \\omega）替换为双反斜杠，
    使其成为合法的 JSON 字符串字面量。仅处理字符串内部的反斜杠。"""
    result: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == '\\' and i + 1 < len(text):
            next_ch = text[i + 1]
            if next_ch not in _JSON_VALID_ESCAPES:
                # 非法转义：把单个 \ 补全为 \\
                result.append('\\\\')
            else:
                result.append('\\')
            i += 1
        else:
            result.append(ch)
            i += 1
    return ''.join(result)


def extract_json_from_llm_response(text: str) -> dict[str, str]:
    """
    从 LLM 的原始输出中稳健地提取 JSON 对象。

    策略（按优先级）：
    1. 直接 json.loads
    2. 修复非法 JSON 转义后再解析（处理含 LaTeX 反斜杠的输出）
    3. 提取 ```json ... ``` 代码块
    4. 正则找第一个 { ... } 块
    5. 抛出 ValueError
    """
    # 1. 直接解析
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. 修复非法 JSON 反斜杠转义后重试（应对 LaTeX 等含 \omega \Sigma 的内容）
    try:
        return json.loads(_fix_invalid_json_escapes(text))
    except json.JSONDecodeError:
        pass

    # 3. 代码块
    code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_block:
        try:
            return json.loads(code_block.group(1))
        except json.JSONDecodeError:
            try:
                return json.loads(_fix_invalid_json_escapes(code_block.group(1)))
            except json.JSONDecodeError:
                pass

    # 4. 首个 { ... } 对象（贪婪，允许嵌套）
    brace_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            try:
                return json.loads(_fix_invalid_json_escapes(brace_match.group(0)))
            except json.JSONDecodeError:
                pass

    raise ValueError(f"无法从 LLM 响应中解析出 JSON。原文片段:\n{text[:500]}")


AVG_CHARS_PER_TOKEN = 4.0   # 英文平均值

def estimate_token_count(text: str) -> int:
    """粗略估计 token 数（无需加载 tiktoken）。"""
    return max(1, int(len(text) / AVG_CHARS_PER_TOKEN))
