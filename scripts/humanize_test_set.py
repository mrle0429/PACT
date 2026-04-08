#!/usr/bin/env python
"""
对 JSONL 中指定文本字段做整段自然化改写的独立脚本。

特点：
- 不复用原有句子级 rewrite prompt
- 直接对整段文本进行整体 humanize
- 模型配置复用 src/config.py 中的 SUPPORTED_MODELS

示例：
  python scripts/humanize_test_set.py
  python scripts/humanize_test_set.py --model qwen3.5-flash
  python scripts/humanize_test_set.py --input-field original_text --output-field original_text_humanized
  python scripts/humanize_test_set.py --limit 10 --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DatasetConfig
from src.utils import (
    AsyncRateLimiter,
    NonRetryableAPIError,
    async_retry,
    get_logger,
    is_non_retryable_api_error,
)

logger = get_logger("humanize_test_set")


def build_humanize_system_prompt(language_hint: str = "English") -> str:
    return f"""You are a professional editor.
Your task is to rewrite the passage in a more natural, fluent, and human-like way while preserving its original meaning, factual content, and approximate length. Do not add new information or remove key details.
"""


def build_humanize_user_prompt(text: str, language_hint: str = "English") -> str:
    return f"""Please humanize the following passage.

Target language: {language_hint}

<text>
{text}
</text>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对 JSONL 文本字段做整段自然化改写，并写回新字段。",
    )
    parser.add_argument(
        "--input-file",
        default=str(PROJECT_ROOT / "data" / "test_set.jsonl"),
        help="输入 JSONL 文件路径。",
    )
    parser.add_argument(
        "--output-file",
        default="",
        help="输出 JSONL 文件路径；为空时自动生成。",
    )
    parser.add_argument(
        "--input-field",
        default="mixed_text",
        help="要改写的输入字段名，默认 mixed_text。",
    )
    parser.add_argument(
        "--output-field",
        default="humanized_text",
        help="写回的输出字段名，默认 humanized_text。",
    )
    parser.add_argument(
        "--model",
        default="qwen3.5-flash",
        help="改写模型名，需存在于 src/config.py 的 SUPPORTED_MODELS。",
    )
    parser.add_argument(
        "--language",
        default="English",
        help="语言提示，默认 English。",
    )
    parser.add_argument(
        "--concurrent-requests",
        type=int,
        default=8,
        help="并发请求数，默认 8。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="仅处理前 N 条记录；0 表示全部处理。",
    )
    parser.add_argument(
        "--only-missing-output",
        action="store_true",
        help="仅处理尚未包含非空输出字段的记录。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="不调用模型，直接把输入字段复制到输出字段，用于检查流程。",
    )
    return parser.parse_args()


def resolve_output_path(input_file: str, output_file: str, model: str) -> Path:
    if output_file:
        return Path(output_file)

    input_path = Path(input_file)
    suffix = input_path.suffix or ".jsonl"
    filename = f"{input_path.stem}.humanized.{model}{suffix}"
    return input_path.with_name(filename)


def load_jsonl(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line_no, raw_line in enumerate(file_obj, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL 解析失败: line={line_no}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"JSONL 行必须是对象: line={line_no}")
            records.append(record)
            if limit > 0 and len(records) >= limit:
                break
    return records


def load_existing_output_ids(path: Path) -> set[str]:
    existing_ids: set[str] = set()
    if not path.exists():
        return existing_ids

    with path.open("r", encoding="utf-8") as file_obj:
        for line_no, raw_line in enumerate(file_obj, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"输出 JSONL 解析失败: line={line_no}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"输出 JSONL 行必须是对象: line={line_no}")
            record_id = record.get("id")
            if isinstance(record_id, str) and record_id.strip():
                existing_ids.add(record_id.strip())

    return existing_ids


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file_obj:
        for record in records:
            file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")
        file_obj.flush()


def build_output_record(
    record: dict[str, Any],
    output_field: str,
    rewritten_text: str,
    model: str,
    input_field: str,
    dry_run: bool,
) -> dict[str, Any]:
    output_record = dict(record)
    output_record[output_field] = rewritten_text

    extra = output_record.get("extra")
    extra_dict = dict(extra) if isinstance(extra, dict) else {}
    extra_dict["humanizer"] = {
        "model": model,
        "input_field": input_field,
        "output_field": output_field,
        "dry_run": dry_run,
        "mode": "full_passage",
        "prompt_version": "v1",
    }
    output_record["extra"] = extra_dict
    return output_record


class HumanizerClient:
    DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    DEEPSEEK_BASE_URL = "https://api.deepseek.com"
    ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
    MINIMAX_BASE_URL = "https://api.minimaxi.com/anthropic"

    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg
        self.model_cfg = cfg.get_model_config()
        self._rate_limiter = AsyncRateLimiter(self.model_cfg.requests_per_minute)
        self._semaphore = asyncio.Semaphore(cfg.concurrent_requests)
        self._client: Any | None = None
        self._provider = self.model_cfg.provider

    async def __aenter__(self) -> "HumanizerClient":
        self._init_client()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

    def _init_client(self) -> None:
        provider = self._provider
        if provider in {"openai", "dashscope", "deepseek", "doubao", "openrouter"}:
            self._init_openai_family_client(provider)
            return
        if provider in {"anthropic", "minimax"}:
            self._init_anthropic_family_client(provider)
            return
        if provider == "gemini":
            self._init_gemini_client()
            return
        if provider == "ollama":
            self._init_ollama_client()
            return
        raise ValueError(f"不支持的 provider: {provider}")

    def _init_openai_family_client(self, provider: str) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError("请安装 openai: pip install openai") from exc

        if provider == "openai":
            if not self.cfg.openai_api_key:
                raise ValueError("未设置 OPENAI_API_KEY 环境变量。")
            self._client = AsyncOpenAI(api_key=self.cfg.openai_api_key)
            return
        if provider == "dashscope":
            if not self.cfg.dashscope_api_key:
                raise ValueError("未设置 DASHSCOPE_API_KEY 环境变量。")
            self._client = AsyncOpenAI(
                api_key=self.cfg.dashscope_api_key,
                base_url=self.DASHSCOPE_BASE_URL,
            )
            return
        if provider == "deepseek":
            if not self.cfg.deepseek_api_key:
                raise ValueError("未设置 DEEPSEEK_API_KEY 环境变量。")
            self._client = AsyncOpenAI(
                api_key=self.cfg.deepseek_api_key,
                base_url=self.DEEPSEEK_BASE_URL,
            )
            return
        if provider == "doubao":
            if not self.cfg.ark_api_key:
                raise ValueError("未设置 ARK_API_KEY 环境变量。")
            self._client = AsyncOpenAI(
                api_key=self.cfg.ark_api_key,
                base_url=self.ARK_BASE_URL,
            )
            return
        if provider == "openrouter":
            if not self.cfg.openrouter_api_key:
                raise ValueError("未设置 OPENROUTER_API_KEY 环境变量。")
            self._client = AsyncOpenAI(
                api_key=self.cfg.openrouter_api_key,
                base_url=self.cfg.openrouter_base_url,
            )
            return

    def _init_anthropic_family_client(self, provider: str) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("请安装 anthropic: pip install anthropic") from exc

        if provider == "anthropic":
            if not self.cfg.anthropic_api_key:
                raise ValueError("未设置 ANTHROPIC_API_KEY 环境变量。")
            self._client = anthropic.AsyncAnthropic(api_key=self.cfg.anthropic_api_key)
            return
        if provider == "minimax":
            if not self.cfg.minimax_api_key:
                raise ValueError("未设置 MINIMAX_API_KEY 环境变量。")
            self._client = anthropic.AsyncAnthropic(
                api_key=self.cfg.minimax_api_key,
                base_url=self.MINIMAX_BASE_URL,
            )
            return

    def _init_gemini_client(self) -> None:
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError("请安装 google-genai: pip install google-genai") from exc
        if not self.cfg.gemini_api_key:
            raise ValueError("未设置 GEMINI_API_KEY 环境变量。")
        self._client = genai.Client(api_key=self.cfg.gemini_api_key)

    def _init_ollama_client(self) -> None:
        try:
            import httpx
        except ImportError as exc:
            raise ImportError("Ollama 接入需要 httpx，请安装: pip install httpx") from exc
        self._client = httpx.AsyncClient(
            base_url=self.cfg.ollama_base_url.rstrip("/"),
            timeout=300.0,
            trust_env=False,
        )

    async def aclose(self) -> None:
        if self._provider == "ollama" and self._client is not None:
            await self._client.aclose()

    async def humanize(
        self,
        text: str,
        language_hint: str,
        task_id: str = "",
    ) -> tuple[str, int, int]:
        clean_text = text.strip()
        if not clean_text:
            return text, 0, 0

        system_prompt = build_humanize_system_prompt(language_hint)
        user_prompt = build_humanize_user_prompt(clean_text, language_hint)

        async with self._semaphore:
            await self._rate_limiter.acquire()
            rewritten_text, input_tokens, output_tokens = await self._call_with_retry(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
            )

        final_text = (rewritten_text or "").strip()
        if not final_text:
            logger.warning("[%s] 模型返回空文本，回退原文。", task_id or "unknown")
            return clean_text, input_tokens, output_tokens

        return final_text, input_tokens, output_tokens

    async def _call_with_retry(
        self,
        user_prompt: str,
        system_prompt: str,
    ) -> tuple[str, int, int]:
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
            return await self._call_api(user_prompt=user_prompt, system_prompt=system_prompt)

        return await _inner()

    async def _call_api(
        self,
        user_prompt: str,
        system_prompt: str,
    ) -> tuple[str, int, int]:
        provider = self._provider
        if provider in {"openai", "dashscope", "deepseek", "doubao", "openrouter"}:
            return await self._call_openai_family(user_prompt, system_prompt, provider)
        if provider in {"anthropic", "minimax"}:
            return await self._call_anthropic_family(user_prompt, system_prompt, provider)
        if provider == "gemini":
            return await self._call_gemini(user_prompt, system_prompt)
        if provider == "ollama":
            return await self._call_ollama(user_prompt, system_prompt)
        raise ValueError(f"不支持的 provider: {provider}")

    async def _call_openai_family(
        self,
        user_prompt: str,
        system_prompt: str,
        provider: str,
    ) -> tuple[str, int, int]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        kwargs: dict[str, Any] = {
            "model": self.model_cfg.model_id,
            "messages": messages,
            "temperature": self.model_cfg.temperature,
            "max_tokens": self.model_cfg.max_output_tokens,
        }

        if provider == "dashscope":
            kwargs["extra_body"] = {"enable_thinking": False}
        elif provider == "doubao":
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        elif provider == "openrouter":
            kwargs["extra_body"] = {
                "reasoning": {
                    "enabled": True,
                    "exclude": True,
                }
            }

        response = await self._client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
        return content, input_tokens, output_tokens

    async def _call_anthropic_family(
        self,
        user_prompt: str,
        system_prompt: str,
        provider: str,
    ) -> tuple[str, int, int]:
        kwargs: dict[str, Any] = {
            "model": self.model_cfg.model_id,
            "max_tokens": self.model_cfg.max_output_tokens,
            "temperature": self.model_cfg.temperature,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }

        if provider == "minimax":
            kwargs["messages"] = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_prompt}],
                }
            ]
            kwargs["thinking"] = {"type": "disabled"}

        response = await self._client.messages.create(**kwargs)

        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
        output_tokens = getattr(usage, "output_tokens", 0) if usage else 0
        return content, input_tokens, output_tokens

    async def _call_gemini(
        self,
        user_prompt: str,
        system_prompt: str,
    ) -> tuple[str, int, int]:
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=self.model_cfg.temperature,
            max_output_tokens=self.model_cfg.max_output_tokens,
            system_instruction=system_prompt,
        )
        response = await self._client.aio.models.generate_content(
            model=self.model_cfg.model_id,
            contents=user_prompt,
            config=config,
        )
        content = response.text or ""
        usage = getattr(response, "usage_metadata", None)
        input_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
        output_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0
        return content, input_tokens, output_tokens

    async def _call_ollama(
        self,
        user_prompt: str,
        system_prompt: str,
    ) -> tuple[str, int, int]:
        payload: dict[str, Any] = {
            "model": self.model_cfg.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "temperature": self.model_cfg.temperature,
                "num_predict": self.model_cfg.max_output_tokens,
            },
            "keep_alive": self.cfg.ollama_keep_alive,
            "stream": False,
        }
        if self.model_cfg.model_id.startswith("gemma4"):
            payload["think"] = False

        response = await self._client.post("/chat", json=payload)
        response.raise_for_status()

        data = response.json()
        if data.get("error"):
            raise RuntimeError(str(data["error"]))

        message = data.get("message") or {}
        content = message.get("content", "") or ""
        input_tokens = int(data.get("prompt_eval_count") or 0)
        output_tokens = int(data.get("eval_count") or 0)
        return content, input_tokens, output_tokens


async def process_record(
    index: int,
    record: dict[str, Any],
    input_field: str,
    output_field: str,
    model: str,
    language: str,
    humanizer: HumanizerClient | None,
    only_missing_output: bool,
    dry_run: bool,
) -> tuple[dict[str, Any], dict[str, int]]:
    existing_output = record.get(output_field)
    if only_missing_output and isinstance(existing_output, str) and existing_output.strip():
        return dict(record), {
            "rewritten": 0,
            "skipped_existing": 1,
            "input_tokens": 0,
            "output_tokens": 0,
        }

    raw_text = record.get(input_field, "")
    text = raw_text if isinstance(raw_text, str) else str(raw_text)
    task_id = str(record.get("id", f"line-{index + 1}"))

    if humanizer is None:
        rewritten_text = text
        input_tokens = 0
        output_tokens = 0
    else:
        rewritten_text, input_tokens, output_tokens = await humanizer.humanize(
            text=text,
            language_hint=language,
            task_id=task_id,
        )

    output_record = build_output_record(
        record=record,
        output_field=output_field,
        rewritten_text=rewritten_text,
        model=model,
        input_field=input_field,
        dry_run=dry_run,
    )
    return output_record, {
        "rewritten": 1,
        "skipped_existing": 0,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


async def run() -> None:
    args = parse_args()
    input_path = Path(args.input_file)
    output_path = resolve_output_path(args.input_file, args.output_file, args.model)

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("输出文件不能与输入文件相同，请改用新的输出路径。")

    records = load_jsonl(input_path, limit=args.limit)
    if not records:
        logger.warning("没有读取到任何记录，退出。")
        return

    existing_output_ids = load_existing_output_ids(output_path)
    pending_records: list[tuple[int, dict[str, Any]]] = []
    skipped_by_output = 0
    for index, record in enumerate(records):
        record_id = record.get("id")
        if isinstance(record_id, str) and record_id.strip() and record_id.strip() in existing_output_ids:
            skipped_by_output += 1
            continue
        pending_records.append((index, record))

    cfg = DatasetConfig(
        rewrite_model=args.model,
        source_path=str(input_path),
        output_dir=str(output_path.parent),
        concurrent_requests=max(1, args.concurrent_requests),
    )

    logger.info(
        "开始整段 humanize: input=%s output=%s model=%s field=%s->%s records=%d dry_run=%s",
        input_path,
        output_path,
        args.model,
        args.input_field,
        args.output_field,
        len(records),
        args.dry_run,
    )
    logger.info(
        "断点对账: 输出文件已有 %d 条已完成记录，本次待处理 %d 条。",
        skipped_by_output,
        len(pending_records),
    )

    if not pending_records:
        logger.info("所有记录都已在最终输出文件中，无需继续处理。")
        return

    stats = {
        "rewritten": 0,
        "skipped_existing": 0,
        "skipped_by_output": skipped_by_output,
        "input_tokens": 0,
        "output_tokens": 0,
    }
    chunk_size = max(1, args.concurrent_requests * 2)

    if args.dry_run:
        humanizer = None
        for start in range(0, len(pending_records), chunk_size):
            chunk = pending_records[start:start + chunk_size]
            results = await asyncio.gather(
                *[
                    process_record(
                        index=index,
                        record=record,
                        input_field=args.input_field,
                        output_field=args.output_field,
                        model=args.model,
                        language=args.language,
                        humanizer=humanizer,
                        only_missing_output=args.only_missing_output,
                        dry_run=args.dry_run,
                    )
                    for index, record in chunk
                ]
            )
            chunk_records: list[dict[str, Any]] = []
            for output_record, item_stats in results:
                chunk_records.append(output_record)
                for key, value in item_stats.items():
                    stats[key] += value
            append_jsonl(output_path, chunk_records)
            logger.info("进度: %d/%d", start + len(chunk), len(pending_records))
    else:
        async with HumanizerClient(cfg) as humanizer:
            for start in range(0, len(pending_records), chunk_size):
                chunk = pending_records[start:start + chunk_size]
                results = await asyncio.gather(
                    *[
                        process_record(
                            index=index,
                            record=record,
                            input_field=args.input_field,
                            output_field=args.output_field,
                            model=args.model,
                            language=args.language,
                            humanizer=humanizer,
                            only_missing_output=args.only_missing_output,
                            dry_run=args.dry_run,
                        )
                        for index, record in chunk
                    ]
                )
                chunk_records: list[dict[str, Any]] = []
                for output_record, item_stats in results:
                    chunk_records.append(output_record)
                    for key, value in item_stats.items():
                        stats[key] += value
                append_jsonl(output_path, chunk_records)
                logger.info("进度: %d/%d", start + len(chunk), len(pending_records))

    logger.info(
        "完成: rewritten=%d skipped_existing=%d skipped_by_output=%d input_tokens=%d output_tokens=%d -> %s",
        stats["rewritten"],
        stats["skipped_existing"],
        stats["skipped_by_output"],
        stats["input_tokens"],
        stats["output_tokens"],
        output_path,
    )


if __name__ == "__main__":
    asyncio.run(run())
