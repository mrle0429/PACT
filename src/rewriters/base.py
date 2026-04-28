"""
LLM 改写公共基类与共享工具。

职责：
- Prompt 构建
- API 调用日志
- 通用重试 / 限速 / 并发控制
- 响应 JSON 解析
"""
from __future__ import annotations

import asyncio
import json
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import DatasetConfig, ModelConfig
from ..sentence_processor import split_into_sentences
from ..utils import (
    AsyncRateLimiter,
    NonRetryableAPIError,
    async_retry,
    extract_json_from_llm_response,
    get_logger,
    is_non_retryable_api_error,
)

logger = get_logger(__name__)


def _build_numbered_context(sentences: list[str]) -> str:
    """将句子列表格式化为带序号的上下文，供改写 Prompt 使用。"""
    return "\n".join(f"[{i + 1}] {sentence}" for i, sentence in enumerate(sentences))


def build_rewrite_system_prompt(language_hint: str = "English") -> str:
    """
    构建稳定的系统提示词。

    这部分尽量保持静态，便于支持前缀缓存的 provider 复用。
    """
    return f"""You are a professional editor.

Your task is to rewrite only the sentences explicitly specified by the user.

Requirements:
- Preserve the original meaning and factual content of each rewritten sentence.
- Make the style and phrasing sound natural and fluent in {language_hint}.
- Do not rewrite any sentence that is not explicitly selected.
- Each rewritten value must remain exactly one sentence.
- Do not output the full article.
- Output only a strict JSON object mapping 1-indexed sentence numbers to rewritten sentence text.
- Do not include any explanation, commentary, markdown, or extra text."""


def build_rewrite_user_prompt(
    sentences: list[str],
    selected_indices: list[int],
) -> str:
    """构建与当前样本相关的用户提示词。"""
    numbered_context = _build_numbered_context(sentences)
    target_indices_str = ", ".join(str(i + 1) for i in selected_indices)
    example_keys = ", ".join(f'"{i + 1}": "<rewritten>"' for i in selected_indices[:2])

    return f"""Rewrite ONLY the following sentence numbers: {target_indices_str}

Article context:
<context>
{numbered_context}
</context>

Return format:
{{{example_keys}, ...}}

Important:
- Return every requested sentence number exactly once.
- Each value must be a single rewritten sentence, not multiple sentences.
- Do not return unrequested sentence numbers."""
    


def build_rewrite_prompt(
    sentences: list[str],
    selected_indices: list[int],
    language_hint: str = "English",
) -> str:
    """
    构建改写 Prompt。

    这里属于 LLM 交互层，而不是句子处理层，因为它描述的是：
    - 如何向模型表达任务
    - 如何约束模型输出格式
    - 如何把句子索引映射到 JSON Diff
    """
    system_prompt = build_rewrite_system_prompt(language_hint)
    user_prompt = build_rewrite_user_prompt(sentences, selected_indices)
    return f"{system_prompt}\n\n{user_prompt}"


class ApiCallLogger:
    """
    将每次 LLM API 调用的完整信息写入 JSONL 文件，便于事后排查。
    """

    def __init__(self, log_dir: str | Path, run_name: str):
        self._dir = Path(log_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / f"{run_name}.jsonl"
        self._lock = threading.Lock()
        logger.info(f"API 调用日志 → {self._path}")

    def log(
        self,
        *,
        task_id: str,
        model: str,
        prompt: str,
        raw_response: str,
        input_tokens: int,
        output_tokens: int,
        parse_ok: bool,
        error: str = "",
    ) -> None:
        record: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task_id": task_id,
            "model": model,
            "prompt": prompt,
            "raw_response": raw_response,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "parse_ok": parse_ok,
            "error": error,
        }
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with self._path.open("a", encoding="utf-8") as file_obj:
                file_obj.write(line + "\n")


class RewriteResult:
    """封装一次 API 改写结果。"""

    __slots__ = (
        "rewrites",
        "model_id",
        "input_tokens",
        "output_tokens",
        "prompt",
        "missing_indices",
        "invalid_indices",
        "extra_indices",
        "error",
    )

    def __init__(
        self,
        rewrites: dict[int, str],
        model_id: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        prompt: str = "",
        missing_indices: list[int] | None = None,
        invalid_indices: list[int] | None = None,
        extra_indices: list[int] | None = None,
        error: str = "",
    ):
        self.rewrites = rewrites
        self.model_id = model_id
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.prompt = prompt
        self.missing_indices = missing_indices or []
        self.invalid_indices = invalid_indices or []
        self.extra_indices = extra_indices or []
        self.error = error


class BaseLLMRewriter(ABC):
    """
    所有 LLM 改写器的统一接口。

    子类只需要关心具体 provider 的 API 调用；
    Prompt 构建、解析、限速、重试都由基类统一处理。
    """

    def __init__(
        self,
        cfg: DatasetConfig,
        model_cfg: ModelConfig,
        api_logger: ApiCallLogger | None = None,
    ):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self._rate_limiter = AsyncRateLimiter(model_cfg.requests_per_minute)
        self._semaphore = asyncio.Semaphore(cfg.concurrent_requests)
        self._api_logger = api_logger

    async def aclose(self) -> None:
        """释放底层客户端资源；默认无操作。"""
        return None

    async def rewrite(
        self,
        sentences: list[str],
        selected_indices: list[int],
        language_hint: str = "English",
        task_id: str = "",
    ) -> RewriteResult:
        """对选中的句子执行改写。"""
        if not selected_indices:
            return RewriteResult({}, self.model_cfg.model_id)

        system_prompt = build_rewrite_system_prompt(language_hint)
        user_prompt = build_rewrite_user_prompt(sentences, selected_indices)
        prompt = f"{system_prompt}\n\n{user_prompt}"

        async with self._semaphore:
            await self._rate_limiter.acquire()
            raw_text, input_tokens, output_tokens = await self._call_api_with_retry(
                user_prompt,
                system_prompt,
            )

        rewrites, diagnostics = self._parse_response(raw_text, selected_indices, task_id=task_id)
        parse_ok = (
            not diagnostics["missing_indices"]
            and not diagnostics["invalid_indices"]
            and not diagnostics["extra_indices"]
        )

        if self._api_logger is not None:
            self._api_logger.log(
                task_id=task_id,
                model=self.model_cfg.model_id,
                prompt=prompt,
                raw_response=raw_text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                parse_ok=parse_ok,
                error="" if parse_ok else diagnostics["error"],
            )

        return RewriteResult(
            rewrites,
            self.model_cfg.model_id,
            input_tokens,
            output_tokens,
            prompt=prompt,
            missing_indices=diagnostics["missing_indices"],
            invalid_indices=diagnostics["invalid_indices"],
            extra_indices=diagnostics["extra_indices"],
            error=diagnostics["error"],
        )

    def _parse_response(
        self,
        raw_text: str,
        selected_indices: list[int],
        task_id: str = "",
    ) -> tuple[dict[int, str], dict[str, Any]]:
        """解析 LLM 返回的 JSON Diff（1-indexed key）为 0-indexed dict。"""
        tag = f"[{task_id}] " if task_id else ""
        requested_keys = {str(index + 1) for index in selected_indices}
        rewrites: dict[int, str] = {}
        missing_indices: list[int] = []
        invalid_indices: list[int] = []
        extra_indices: list[int] = []
        errors: list[str] = []

        try:
            parsed = extract_json_from_llm_response(raw_text)
        except ValueError as exc:
            error = (
                f"JSON 解析失败: {exc}\n"
                f"  模型: {self.model_cfg.model_id} | "
                f"  响应长度: {len(raw_text)} 字符\n"
                f"  响应前 500 字符: {raw_text[:500]}\n"
                f"  响应后 200 字符: {raw_text[-200:]}"
            )
            logger.warning("%s%s", tag, error)
            return {}, {
                "missing_indices": list(selected_indices),
                "invalid_indices": list(selected_indices),
                "extra_indices": [],
                "error": str(exc),
            }

        if not isinstance(parsed, dict):
            error = f"响应 JSON 顶层必须是对象，实际为 {type(parsed).__name__}"
            logger.warning("%s%s", tag, error)
            return {}, {
                "missing_indices": list(selected_indices),
                "invalid_indices": list(selected_indices),
                "extra_indices": [],
                "error": error,
            }

        for key, value in parsed.items():
            key_str = str(key).strip()
            if key_str not in requested_keys:
                try:
                    extra_indices.append(int(key_str) - 1)
                except ValueError:
                    pass
                continue
            if not isinstance(value, str) or not value.strip():
                invalid_indices.append(int(key_str) - 1)
                errors.append(f"key={key_str} 非字符串或为空")
                continue
            rewritten = value.strip()
            rewritten_sentences = split_into_sentences(rewritten)
            if len(rewritten_sentences) != 1:
                invalid_indices.append(int(key_str) - 1)
                errors.append(f"key={key_str} 改写后被切分为 {len(rewritten_sentences)} 句")
                continue
            rewrites[int(key_str) - 1] = rewritten

        for index in selected_indices:
            if index not in rewrites and index not in invalid_indices:
                missing_indices.append(index)

        if missing_indices:
            errors.append(
                "缺失 key=" + ",".join(str(index + 1) for index in missing_indices)
            )
        if extra_indices:
            errors.append(
                "越权 key=" + ",".join(str(index + 1) for index in extra_indices)
            )

        if errors:
            logger.warning("%s部分句子回退原句: %s", tag, " | ".join(errors))

        return rewrites, {
            "missing_indices": missing_indices,
            "invalid_indices": sorted(set(invalid_indices)),
            "extra_indices": sorted(set(extra_indices)),
            "error": "; ".join(errors),
        }

    async def _call_api_with_retry(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
    ) -> tuple[str, int, int]:
        """返回 (响应文本, input_tokens, output_tokens)。"""

        @async_retry(
            max_attempts=self.cfg.max_retries,
            wait_seconds=self.cfg.retry_wait_seconds,
            exceptions=(Exception,),
            should_retry=lambda exc: (
                not isinstance(exc, NonRetryableAPIError)
                and (not is_non_retryable_api_error(exc))
            ),
        )
        async def _inner() -> tuple[str, int, int]:
            return await self._call_api(user_prompt, system_prompt)

        return await _inner()

    @abstractmethod
    async def _call_api(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
    ) -> tuple[str, int, int]:
        """调用底层 LLM API。"""
        ...
