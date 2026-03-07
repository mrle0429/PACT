"""
多模型 LLM 改写模块。

架构：
  BaseLLMRewriter (抽象基类)
    ├── OpenAIRewriter    — openai 库 (AsyncOpenAI)
    ├── AnthropicRewriter — anthropic 库 (AsyncAnthropic)
    ├── GeminiRewriter    — google-genai 库
    ├── MiniMaxRewriter   — anthropic 库 (Anthropic API 兼容, 自定义 base_url)
    └── DashScopeRewriter — openai 库 (OpenAI API 兼容, DashScope base_url)
        
对外唯一接口:
  rewriter.rewrite(sentences, selected_indices) -> dict[int, str]
  dict 的 key 为 0-indexed 句子下标，value 为改写后的句子文本。
"""
from __future__ import annotations

import asyncio
import json
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import DatasetConfig, ModelConfig
from .sentence_processor import build_rewrite_prompt
from .utils import (
    AsyncRateLimiter,
    is_non_retryable_api_error,
    NonRetryableAPIError,
    async_retry,
    extract_json_from_llm_response,
    get_logger,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# API 调用日志记录器（JSONL 格式，逐条追加）
# ---------------------------------------------------------------------------

class ApiCallLogger:
    """
    将每次 LLM API 调用的完整信息写入 JSONL 文件，便于事后排查。

    每条记录包含：
        timestamp, task_id, model, prompt, raw_response,
        input_tokens, output_tokens, parse_ok, error

    线程安全 + 协程安全（使用 threading.Lock 保护文件写入）。
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
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")


# ---------------------------------------------------------------------------
# 改写结果数据类
# ---------------------------------------------------------------------------

class RewriteResult:
    """
    封装一次 API 改写的返回结果。
    rewrites: {0-indexed 下标: 改写文本}
    """
    __slots__ = ("rewrites", "model_id", "input_tokens", "output_tokens")

    def __init__(
        self,
        rewrites: dict[int, str],
        model_id: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ):
        self.rewrites = rewrites
        self.model_id = model_id
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------

class BaseLLMRewriter(ABC):
    """
    所有 LLM 改写器的统一接口。

    子类须实现 `_call_api`，其余流程（Prompt 构建、JSON 解析、重试、限速）
    由基类统一处理，避免重复。
    """

    def __init__(self, cfg: DatasetConfig, model_cfg: ModelConfig,
                 api_logger: ApiCallLogger | None = None):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self._rate_limiter = AsyncRateLimiter(model_cfg.requests_per_minute)
        self._semaphore = asyncio.Semaphore(cfg.concurrent_requests)
        self._api_logger = api_logger

    # ------------------------------------------------------------------
    # 公共入口
    # ------------------------------------------------------------------

    async def rewrite(
        self,
        sentences: list[str],
        selected_indices: list[int],
        language_hint: str = "English",
        task_id: str = "",
    ) -> RewriteResult:
        """
        对 sentences 中由 selected_indices 指定的句子进行 AI 改写。

        - ratio = 0.0 时直接返回空字典（调用方应在上层跳过）
        - selected_indices 为空时同样返回空字典

        Args:
            task_id: 可选的任务标识符，用于日志定位问题

        Returns:
            RewriteResult，其中 rewrites key 为 0-indexed 下标
        """
        if not selected_indices:
            return RewriteResult({}, self.model_cfg.model_id)

        prompt = build_rewrite_prompt(sentences, selected_indices, language_hint)

        async with self._semaphore:
            await self._rate_limiter.acquire()
            raw_text, in_tok, out_tok = await self._call_api_with_retry(prompt)

        rewrites = self._parse_response(raw_text, selected_indices, task_id=task_id)
        parse_ok = len(rewrites) > 0

        # ---- 记录 API 调用日志 ----
        if self._api_logger is not None:
            self._api_logger.log(
                task_id=task_id,
                model=self.model_cfg.model_id,
                prompt=prompt,
                raw_response=raw_text,
                input_tokens=in_tok,
                output_tokens=out_tok,
                parse_ok=parse_ok,
                error="" if parse_ok else "JSON parse failed or empty rewrites",
            )

        return RewriteResult(rewrites, self.model_cfg.model_id, in_tok, out_tok)

    # ------------------------------------------------------------------
    # 内部：解析 & 验证 API 返回的 JSON Diff
    # ------------------------------------------------------------------

    def _parse_response(
        self,
        raw_text: str,
        selected_indices: list[int],
        task_id: str = "",
    ) -> dict[int, str]:
        """
        解析 LLM 返回的 JSON Diff（1-indexed key）→ 0-indexed dict。
        仅接受在 selected_indices 范围内的 key，其余忽略。
        """
        tag = f"[{task_id}] " if task_id else ""
        try:
            parsed = extract_json_from_llm_response(raw_text)
        except ValueError as exc:
            logger.warning(
                f"{tag}JSON 解析失败，本次改写将被跳过: {exc}\n"
                f"  模型: {self.model_cfg.model_id} | "
                f"  响应长度: {len(raw_text)} 字符\n"
                f"  响应前 500 字符: {raw_text[:500]}\n"
                f"  响应后 200 字符: {raw_text[-200:]}"
            )
            return {}

        allowed_one_indexed = {str(i + 1) for i in selected_indices}
        rewrites: dict[int, str] = {}
        for key, value in parsed.items():
            key_str = str(key).strip()
            if key_str not in allowed_one_indexed:
                logger.debug(f"忽略越权 key={key_str}（不在选中句子范围内）")
                continue
            if not isinstance(value, str) or not value.strip():
                logger.debug(f"忽略空值 key={key_str}")
                continue
            rewrites[int(key_str) - 1] = value.strip()

        # 检查覆盖率
        expected = len(selected_indices)
        got = len(rewrites)
        if got < expected:
            logger.warning(
                f"改写覆盖不完整: 请求 {expected} 句，仅返回 {got} 句。"
                f"缺失 key: {set(str(i+1) for i in selected_indices) - set(str(k+1) for k in rewrites)}"
            )
        return rewrites

    # ------------------------------------------------------------------
    # 内部：带重试的 API 调用
    # ------------------------------------------------------------------

    async def _call_api_with_retry(
        self, prompt: str
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
            return await self._call_api(prompt)
        return await _inner()

    # ------------------------------------------------------------------
    # 子类实现
    # ------------------------------------------------------------------

    @abstractmethod
    async def _call_api(self, prompt: str) -> tuple[str, int, int]:
        """
        调用底层 LLM API。

        Returns:
            (响应文本, input_tokens, output_tokens)
        """
        ...


# ---------------------------------------------------------------------------
# OpenAI 实现（支持 gpt-4o-mini, gpt-4o 等）
# ---------------------------------------------------------------------------

class OpenAIRewriter(BaseLLMRewriter):
    """使用 openai 官方异步客户端。"""

    def __init__(self, cfg: DatasetConfig, model_cfg: ModelConfig,
                 api_logger: ApiCallLogger | None = None):
        super().__init__(cfg, model_cfg, api_logger)
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError("请安装 openai: pip install openai") from exc
        if not cfg.openai_api_key:
            raise ValueError("未设置 OPENAI_API_KEY 环境变量。")
        self._client = AsyncOpenAI(api_key=cfg.openai_api_key)

    async def _call_api(self, prompt: str) -> tuple[str, int, int]:
        response = await self._client.chat.completions.create(
            model=self.model_cfg.model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.model_cfg.temperature,
            max_tokens=self.model_cfg.max_output_tokens,
        )
        content = response.choices[0].message.content or ""
        usage = response.usage
        in_tok = usage.prompt_tokens if usage else 0
        out_tok = usage.completion_tokens if usage else 0
        return content, in_tok, out_tok


# ---------------------------------------------------------------------------
# Anthropic 实现（支持 claude-3-5-haiku, claude-3-5-sonnet 等）
# ---------------------------------------------------------------------------

class AnthropicRewriter(BaseLLMRewriter):
    """使用 anthropic 官方异步客户端。"""

    def __init__(self, cfg: DatasetConfig, model_cfg: ModelConfig,
                 api_logger: ApiCallLogger | None = None):
        super().__init__(cfg, model_cfg, api_logger)
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("请安装 anthropic: pip install anthropic") from exc
        if not cfg.anthropic_api_key:
            raise ValueError("未设置 ANTHROPIC_API_KEY 环境变量。")
        self._client = anthropic.AsyncAnthropic(api_key=cfg.anthropic_api_key)

    async def _call_api(self, prompt: str) -> tuple[str, int, int]:
        response = await self._client.messages.create(
            model=self.model_cfg.model_id,
            max_tokens=self.model_cfg.max_output_tokens,
            temperature=self.model_cfg.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text
        in_tok = response.usage.input_tokens if response.usage else 0
        out_tok = response.usage.output_tokens if response.usage else 0
        return content, in_tok, out_tok


# ---------------------------------------------------------------------------
# Google Gemini 实现（支持 gemini-2.0-flash / gemini-3.x 等）
# ---------------------------------------------------------------------------

class GeminiRewriter(BaseLLMRewriter):
    """使用 google-genai 库（新版 SDK，原生 async）。"""

    def __init__(self, cfg: DatasetConfig, model_cfg: ModelConfig,
                 api_logger: ApiCallLogger | None = None):
        super().__init__(cfg, model_cfg, api_logger)
        try:
            from google import genai
            self._genai = genai
        except ImportError as exc:
            raise ImportError(
                "请安装 google-genai: pip install google-genai"
            ) from exc
        if not cfg.gemini_api_key:
            raise ValueError("未设置 GEMINI_API_KEY 环境变量。")
        self._client = genai.Client(api_key=cfg.gemini_api_key)

    async def _call_api(self, prompt: str) -> tuple[str, int, int]:
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=self.model_cfg.temperature,
            max_output_tokens=self.model_cfg.max_output_tokens,
        )
        response = await self._client.aio.models.generate_content(
            model=self.model_cfg.model_id,
            contents=prompt,
            config=config,
        )
        content = response.text or ""
        # usage_metadata
        usage = getattr(response, "usage_metadata", None)
        in_tok = getattr(usage, "prompt_token_count", 0) if usage else 0
        out_tok = getattr(usage, "candidates_token_count", 0) if usage else 0
        return content, in_tok, out_tok


# ---------------------------------------------------------------------------
# MiniMax 实现（Anthropic API 兼容，自定义 base_url）
# ---------------------------------------------------------------------------

class MiniMaxRewriter(BaseLLMRewriter):
    """
    使用 Anthropic SDK 调用 MiniMax 模型。

    MiniMax 提供了 Anthropic API 兼容接口，
    只需将 base_url 指向 https://api.minimaxi.com/anthropic 即可。
    """

    MINIMAX_BASE_URL = "https://api.minimaxi.com/anthropic"

    def __init__(self, cfg: DatasetConfig, model_cfg: ModelConfig,
                 api_logger: ApiCallLogger | None = None):
        super().__init__(cfg, model_cfg, api_logger)
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("MiniMax 使用 Anthropic SDK，请安装: pip install anthropic") from exc
        if not cfg.minimax_api_key:
            raise ValueError("未设置 MINIMAX_API_KEY 环境变量。")
        self._client = anthropic.AsyncAnthropic(
            api_key=cfg.minimax_api_key,
            base_url=self.MINIMAX_BASE_URL,
        )

    async def _call_api(self, prompt: str) -> tuple[str, int, int]:
        response = await self._client.messages.create(
            model=self.model_cfg.model_id,
            max_tokens=self.model_cfg.max_output_tokens,
            temperature=self.model_cfg.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text
        in_tok = response.usage.input_tokens if response.usage else 0
        out_tok = response.usage.output_tokens if response.usage else 0
        return content, in_tok, out_tok


# ---------------------------------------------------------------------------
# DashScope / Qwen 实现（OpenAI API 兼容，自定义 base_url）
# ---------------------------------------------------------------------------

class DashScopeRewriter(BaseLLMRewriter):
    """
    使用 OpenAI SDK 调用阿里云百炼 DashScope 模型（qwen 系列）。

    DashScope 提供 OpenAI 兼容接口，
    base_url 为 https://dashscope.aliyuncs.com/compatible-mode/v1。
    """

    DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    _JSON_SCHEMA_RESPONSE_FORMAT: dict[str, Any] = {
        "type": "json_schema",
        "json_schema": {
            "name": "rewrites_by_sentence_index",
            "description": "Map from 1-indexed sentence number to rewritten sentence text.",
            "schema": {
                "type": "object",
                "propertyNames": {"pattern": "^[1-9][0-9]*$"},
                "additionalProperties": {"type": "string", "minLength": 1},
            },
            "strict": True,
        },
    }
    _JSON_OBJECT_RESPONSE_FORMAT: dict[str, str] = {"type": "json_object"}

    def __init__(self, cfg: DatasetConfig, model_cfg: ModelConfig,
                 api_logger: ApiCallLogger | None = None):
        super().__init__(cfg, model_cfg, api_logger)
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError("DashScope 使用 OpenAI SDK，请安装: pip install openai") from exc
        if not cfg.dashscope_api_key:
            raise ValueError("未设置 DASHSCOPE_API_KEY 环境变量。")
        self._client = AsyncOpenAI(
            api_key=cfg.dashscope_api_key,
            base_url=self.DASHSCOPE_BASE_URL,
        )

    async def _call_api(self, prompt: str) -> tuple[str, int, int]:
        # 优先使用 json_schema（strict=true）提升结构化输出稳定性；
        # 若服务端/模型不支持，则回退到 json_object。
        try:
            response = await self._client.chat.completions.create(
                model=self.model_cfg.model_id,
                messages=[{"role": "user", "content": prompt}],
                extra_body={"enable_thinking": False},  # 关闭百炼特有的“思考”功能，减少无关输出
                temperature=self.model_cfg.temperature,
                max_tokens=self.model_cfg.max_output_tokens,
                response_format=self._JSON_SCHEMA_RESPONSE_FORMAT,
            )
        except Exception as exc:
            err = str(exc).lower()
            if ("response_format" in err) or ("json_schema" in err) or ("unsupported" in err):
                logger.warning(
                    "DashScope json_schema response_format 不可用，回退到 json_object。"
                )
                response = await self._client.chat.completions.create(
                    model=self.model_cfg.model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.model_cfg.temperature,
                    max_tokens=self.model_cfg.max_output_tokens,
                    response_format=self._JSON_OBJECT_RESPONSE_FORMAT,
                )
            else:
                raise
        content = response.choices[0].message.content or ""
        usage = response.usage
        in_tok = usage.prompt_tokens if usage else 0
        out_tok = usage.completion_tokens if usage else 0
        return content, in_tok, out_tok


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------

def create_rewriter(
    cfg: DatasetConfig,
    api_logger: ApiCallLogger | None = None,
) -> BaseLLMRewriter:
    """
    根据 cfg.rewrite_model 自动实例化对应的改写器。

    Args:
        api_logger: 可选的 API 调用日志记录器。如果未提供，则自动创建一个
                    写入 output/api_logs/{model}.jsonl 的记录器。
    """
    model_cfg = cfg.get_model_config()
    provider = model_cfg.provider

    # 默认自动创建 API 日志
    if api_logger is None:
        log_dir = Path(cfg.output_dir) / "api_logs"
        run_name = f"{cfg.rewrite_model}_{cfg.source_tag}"
        api_logger = ApiCallLogger(log_dir, run_name)

    if provider == "openai":
        return OpenAIRewriter(cfg, model_cfg, api_logger)
    elif provider == "anthropic":
        return AnthropicRewriter(cfg, model_cfg, api_logger)
    elif provider == "gemini":
        return GeminiRewriter(cfg, model_cfg, api_logger)
    elif provider == "minimax":
        return MiniMaxRewriter(cfg, model_cfg, api_logger)
    elif provider == "dashscope":
        return DashScopeRewriter(cfg, model_cfg, api_logger)
    else:
        raise ValueError(f"未知 provider: '{provider}'")
