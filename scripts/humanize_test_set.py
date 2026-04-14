#!/usr/bin/env python
"""
对 `test_set.jsonl` 的 `mixed_text` 执行句子级 selective humanize。

规则：
- 仅改写 `sentence_labels == 1` 的句子
- `sentence_labels == 0` 的句子保持不变
- 输出文本与输入文本保持相同句子数
- 模型可配置，复用仓库现有的 provider / model 配置

示例：
  python scripts/humanize_test_set.py
  python scripts/humanize_test_set.py --model qwen3.5-flash
  python scripts/humanize_test_set.py --output-field mixed_text
  python scripts/humanize_test_set.py --limit 10 --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.config import DatasetConfig
except ModuleNotFoundError as exc:
    if exc.name != "dotenv":
        raise
    dotenv_stub = types.ModuleType("dotenv")

    def _load_dotenv_fallback(*_args: Any, **_kwargs: Any) -> bool:
        env_path = PROJECT_ROOT / ".env"
        if not env_path.exists():
            return False

        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key or key in os.environ:
                continue
            if value and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1]
            os.environ[key] = value
        return True

    dotenv_stub.load_dotenv = _load_dotenv_fallback
    sys.modules["dotenv"] = dotenv_stub
    from src.config import DatasetConfig

from src.rewriters import ApiCallLogger, create_rewriter
from src.sentence_processor import split_into_sentences
from src.utils import extract_json_from_llm_response, get_logger

logger = get_logger("humanize_test_set")


def build_humanize_system_prompt(language_hint: str = "English") -> str:
    return f"""You are a professional editor.

Your task is to humanize only the sentences explicitly selected by the user.

Requirements:
- Preserve the original meaning, factual content, and approximate length.
- Keep the writing natural and fluent in {language_hint}.
- Rewrite only the selected sentences.
- Each rewritten value must remain exactly one sentence.
- Do not split, merge, drop, or reorder sentences.
- Return only a strict JSON object that maps 1-indexed sentence numbers to rewritten sentence text.
- Do not include any explanation, commentary, markdown, or extra text."""


def build_humanize_user_prompt(sentences: list[str], selected_indices: list[int]) -> str:
    numbered_context = "\n".join(
        f"[{index + 1}] {sentence}" for index, sentence in enumerate(sentences)
    )
    target_indices_str = ", ".join(str(index + 1) for index in selected_indices)
    example_keys = ", ".join(f'"{index + 1}": "<humanized sentence>"' for index in selected_indices[:2])

    return f"""Humanize ONLY the following sentence numbers: {target_indices_str}

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对 JSONL 中的 mixed_text 做 selective sentence humanize。",
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
        help="输入文本字段，默认 mixed_text。",
    )
    parser.add_argument(
        "--output-field",
        default="mixed_text_humanized",
        help="输出文本字段，默认 mixed_text_humanized。",
    )
    parser.add_argument(
        "--labels-field",
        default="sentence_labels",
        help="句子标签字段，默认 sentence_labels。",
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
        help="仅处理输入记录中输出字段为空的样本。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="不调用模型，直接输出原文，用于检查流程与输出结构。",
    )
    parser.add_argument(
        "--fail-on-count-mismatch",
        action="store_true",
        help="若 mixed_text 分句数与 sentence_labels 长度不一致则立即报错；默认仅记录并回退原文。",
    )
    return parser.parse_args()


def resolve_output_path(input_file: str, output_file: str, model: str) -> Path:
    if output_file:
        return Path(output_file)

    input_path = Path(input_file)
    suffix = input_path.suffix or ".jsonl"
    filename = f"{input_path.stem}.sentence_humanized.{model}{suffix}"
    return input_path.with_name(filename)


def resolve_failure_log_path(output_path: Path) -> Path:
    suffix = output_path.suffix or ".jsonl"
    filename = f"{output_path.stem}.failures{suffix}"
    return output_path.with_name(filename)


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


def normalize_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return str(value)


def normalize_labels(value: Any, field_name: str, task_id: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"[{task_id}] {field_name} 必须是 list，实际为 {type(value).__name__}")

    normalized: list[int] = []
    for index, item in enumerate(value):
        try:
            label = int(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"[{task_id}] {field_name}[{index}] 无法转换为整数: {item!r}"
            ) from exc
        if label not in {0, 1}:
            raise ValueError(f"[{task_id}] {field_name}[{index}] 必须为 0/1，实际为 {label}")
        normalized.append(label)
    return normalized


def build_output_record(
    record: dict[str, Any],
    *,
    input_field: str,
    output_field: str,
    final_text: str,
    model: str,
    dry_run: bool,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    output_record = dict(record)
    output_record[output_field] = final_text

    extra = output_record.get("extra")
    extra_dict = dict(extra) if isinstance(extra, dict) else {}
    extra_dict["sentence_humanizer"] = {
        "model": model,
        "input_field": input_field,
        "output_field": output_field,
        "dry_run": dry_run,
        "mode": "selected_sentences_only",
        "prompt_version": "v1",
        **metadata,
    }
    output_record["extra"] = extra_dict
    return output_record


def build_failure_record(
    *,
    record: dict[str, Any],
    task_id: str,
    line_index: int,
    model: str,
    input_field: str,
    output_field: str,
    labels_field: str,
    status: str,
    reason: str,
    retryable: bool,
    sentence_count: int,
    label_count: int,
    selected_indices: list[int],
    missing_indices: list[int],
    invalid_indices: list[int],
    extra_indices: list[int],
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> dict[str, Any]:
    return {
        "id": task_id,
        "line_index": line_index,
        "status": status,
        "reason": reason,
        "retryable": retryable,
        "model": model,
        "input_field": input_field,
        "output_field": output_field,
        "labels_field": labels_field,
        "sentence_count": sentence_count,
        "label_count": label_count,
        "selected_sentence_count": len(selected_indices),
        "selected_indices": [index + 1 for index in selected_indices],
        "missing_indices": [index + 1 for index in missing_indices],
        "invalid_indices": [index + 1 for index in invalid_indices],
        "extra_indices": [index + 1 for index in extra_indices],
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "source_record": {
            "id": record.get("id"),
            "source_dataset": record.get("source_dataset"),
            "source_domain": record.get("source_domain"),
        },
    }


class SentenceHumanizer:
    def __init__(self, cfg: DatasetConfig, api_logger: ApiCallLogger):
        self._rewriter = create_rewriter(cfg, api_logger=api_logger)
        self._api_logger = api_logger
        self._model_id = cfg.get_model_config().model_id

    async def __aenter__(self) -> "SentenceHumanizer":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._rewriter.aclose()

    async def humanize_selected(
        self,
        *,
        sentences: list[str],
        selected_indices: list[int],
        language_hint: str,
        task_id: str,
    ) -> tuple[dict[int, str], dict[str, Any]]:
        if not selected_indices:
            return {}, {
                "input_tokens": 0,
                "output_tokens": 0,
                "missing_indices": [],
                "invalid_indices": [],
                "extra_indices": [],
                "raw_response": "",
                "error": "",
            }

        system_prompt = build_humanize_system_prompt(language_hint)
        user_prompt = build_humanize_user_prompt(sentences, selected_indices)
        prompt = f"{system_prompt}\n\n{user_prompt}"

        raw_text, input_tokens, output_tokens = await self._rewriter._call_api_with_retry(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        )

        rewrites, diagnostics = parse_humanizer_response(
            raw_text=raw_text,
            selected_indices=selected_indices,
            task_id=task_id,
        )
        parse_ok = (
            not diagnostics["missing_indices"]
            and not diagnostics["invalid_indices"]
            and not diagnostics["extra_indices"]
        )

        self._api_logger.log(
            task_id=task_id,
            model=self._model_id,
            prompt=prompt,
            raw_response=raw_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            parse_ok=parse_ok,
            error="" if parse_ok else diagnostics["error"],
        )

        return rewrites, {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "missing_indices": diagnostics["missing_indices"],
            "invalid_indices": diagnostics["invalid_indices"],
            "extra_indices": diagnostics["extra_indices"],
            "raw_response": raw_text,
            "error": diagnostics["error"],
        }


def parse_humanizer_response(
    *,
    raw_text: str,
    selected_indices: list[int],
    task_id: str,
) -> tuple[dict[int, str], dict[str, Any]]:
    requested_keys = {str(index + 1) for index in selected_indices}
    rewrites: dict[int, str] = {}
    missing_indices: list[int] = []
    invalid_indices: list[int] = []
    extra_indices: list[int] = []
    errors: list[str] = []

    try:
        payload = extract_json_from_llm_response(raw_text)
    except ValueError as exc:
        errors.append(str(exc))
        logger.warning("[%s] JSON 解析失败，回退原句。", task_id)
        return {}, {
            "missing_indices": list(selected_indices),
            "invalid_indices": list(selected_indices),
            "extra_indices": [],
            "error": "; ".join(errors),
        }

    if not isinstance(payload, dict):
        error = f"响应 JSON 顶层必须是对象，实际为 {type(payload).__name__}"
        logger.warning("[%s] %s，回退原句。", task_id, error)
        return {}, {
            "missing_indices": list(selected_indices),
            "invalid_indices": list(selected_indices),
            "extra_indices": [],
            "error": error,
        }

    for key, value in payload.items():
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
        logger.warning("[%s] 部分句子回退原句: %s", task_id, " | ".join(errors))

    return rewrites, {
        "missing_indices": missing_indices,
        "invalid_indices": sorted(set(invalid_indices)),
        "extra_indices": sorted(set(extra_indices)),
        "error": "; ".join(errors),
    }


async def process_record(
    *,
    index: int,
    record: dict[str, Any],
    input_field: str,
    output_field: str,
    labels_field: str,
    model: str,
    language: str,
    humanizer: SentenceHumanizer | None,
    only_missing_output: bool,
    dry_run: bool,
    fail_on_count_mismatch: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, int]]:
    existing_output = record.get(output_field)
    if only_missing_output and isinstance(existing_output, str) and existing_output.strip():
        return None, None, {
            "written_records": 0,
            "written_selected_sentences": 0,
            "deferred_records": 0,
            "deferred_selected_sentences": 0,
            "invalid_response_records": 0,
            "input_validation_failed_records": 0,
            "post_check_failed_records": 0,
            "skipped_existing": 1,
            "input_tokens": 0,
            "output_tokens": 0,
        }

    task_id = str(record.get("id", f"line-{index + 1}"))
    text = normalize_text(record.get(input_field))
    try:
        labels = normalize_labels(record.get(labels_field), labels_field, task_id)
    except ValueError as exc:
        logger.warning("%s 不写入结果集，本条待处理。", exc)
        failure_record = build_failure_record(
            record=record,
            task_id=task_id,
            line_index=index,
            model=model,
            input_field=input_field,
            output_field=output_field,
            labels_field=labels_field,
            status="invalid_labels",
            reason=str(exc),
            retryable=False,
            sentence_count=len(split_into_sentences(text)),
            label_count=0,
            selected_indices=[],
            missing_indices=[],
            invalid_indices=[],
            extra_indices=[],
        )
        return None, failure_record, {
            "written_records": 0,
            "written_selected_sentences": 0,
            "deferred_records": 1,
            "deferred_selected_sentences": 0,
            "invalid_response_records": 0,
            "input_validation_failed_records": 1,
            "post_check_failed_records": 0,
            "skipped_existing": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }
    sentences = split_into_sentences(text)

    if len(sentences) != len(labels):
        message = (
            f"[{task_id}] {input_field} 分句数({len(sentences)}) 与 "
            f"{labels_field} 长度({len(labels)}) 不一致。"
        )
        if fail_on_count_mismatch:
            raise ValueError(message)
        logger.warning("%s 不写入结果集，本条待处理。", message)
        failure_record = build_failure_record(
            record=record,
            task_id=task_id,
            line_index=index,
            model=model,
            input_field=input_field,
            output_field=output_field,
            labels_field=labels_field,
            status="input_count_mismatch",
            reason=message,
            retryable=False,
            sentence_count=len(sentences),
            label_count=len(labels),
            selected_indices=[],
            missing_indices=[],
            invalid_indices=[],
            extra_indices=[],
        )
        return None, failure_record, {
            "written_records": 0,
            "written_selected_sentences": 0,
            "deferred_records": 1,
            "deferred_selected_sentences": 0,
            "invalid_response_records": 0,
            "input_validation_failed_records": 1,
            "post_check_failed_records": 0,
            "skipped_existing": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }

    selected_indices = [idx for idx, label in enumerate(labels) if label == 1]
    final_sentences = list(sentences)
    missing_indices: list[int] = []
    invalid_indices: list[int] = []
    extra_indices: list[int] = []
    input_tokens = 0
    output_tokens = 0

    if selected_indices and humanizer is not None:
        rewrites, diagnostics = await humanizer.humanize_selected(
            sentences=sentences,
            selected_indices=selected_indices,
            language_hint=language,
            task_id=task_id,
        )
        for idx, rewritten in rewrites.items():
            final_sentences[idx] = rewritten
        input_tokens = int(diagnostics["input_tokens"])
        output_tokens = int(diagnostics["output_tokens"])
        missing_indices = list(diagnostics["missing_indices"])
        invalid_indices = list(diagnostics["invalid_indices"])
        extra_indices = list(diagnostics["extra_indices"])
        if missing_indices or invalid_indices or extra_indices:
            failure_record = build_failure_record(
                record=record,
                task_id=task_id,
                line_index=index,
                model=model,
                input_field=input_field,
                output_field=output_field,
                labels_field=labels_field,
                status="invalid_model_response",
                reason=diagnostics.get("error", ""),
                retryable=True,
                sentence_count=len(sentences),
                label_count=len(labels),
                selected_indices=selected_indices,
                missing_indices=missing_indices,
                invalid_indices=invalid_indices,
                extra_indices=extra_indices,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            return None, failure_record, {
                "written_records": 0,
                "written_selected_sentences": 0,
                "deferred_records": 1,
                "deferred_selected_sentences": len(selected_indices),
                "invalid_response_records": 1,
                "input_validation_failed_records": 0,
                "post_check_failed_records": 0,
                "skipped_existing": 0,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }

    final_text = " ".join(final_sentences)
    final_sentence_count = len(split_into_sentences(final_text))
    status = "ok"

    if final_sentence_count != len(sentences):
        reason = (
            f"[{task_id}] 输出句子数从 {len(sentences)} 变为 {final_sentence_count}，"
            "不写入结果集。"
        )
        logger.warning(reason)
        failure_record = build_failure_record(
            record=record,
            task_id=task_id,
            line_index=index,
            model=model,
            input_field=input_field,
            output_field=output_field,
            labels_field=labels_field,
            status="post_join_count_mismatch",
            reason=reason,
            retryable=True,
            sentence_count=len(sentences),
            label_count=len(labels),
            selected_indices=selected_indices,
            missing_indices=[],
            invalid_indices=[],
            extra_indices=[],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        return None, failure_record, {
            "written_records": 0,
            "written_selected_sentences": 0,
            "deferred_records": 1,
            "deferred_selected_sentences": len(selected_indices),
            "invalid_response_records": 0,
            "input_validation_failed_records": 0,
            "post_check_failed_records": 1,
            "skipped_existing": 0,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

    rewritten_sentence_count = 0 if dry_run else len(selected_indices)
    if not selected_indices:
        status = "no_selected_sentences"
    elif dry_run:
        status = "dry_run"

    output_record = build_output_record(
        record,
        input_field=input_field,
        output_field=output_field,
        final_text=final_text,
        model=model,
        dry_run=dry_run,
        metadata={
            "status": status,
            "sentence_count": len(sentences),
            "label_count": len(labels),
            "selected_sentence_count": len(selected_indices),
            "rewritten_sentence_count": rewritten_sentence_count,
            "fallback_sentence_count": 0,
            "missing_indices": [],
            "invalid_indices": [],
            "extra_indices": [],
        },
    )

    return output_record, None, {
        "written_records": 1,
        "written_selected_sentences": rewritten_sentence_count,
        "deferred_records": 0,
        "deferred_selected_sentences": 0,
        "invalid_response_records": 0,
        "input_validation_failed_records": 0,
        "post_check_failed_records": 0,
        "skipped_existing": 0,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


async def run() -> None:
    args = parse_args()
    input_path = Path(args.input_file)
    output_path = resolve_output_path(args.input_file, args.output_file, args.model)
    failure_log_path = resolve_failure_log_path(output_path)

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
        "开始 selective sentence humanize: input=%s output=%s model=%s field=%s->%s labels=%s records=%d dry_run=%s",
        input_path,
        output_path,
        args.model,
        args.input_field,
        args.output_field,
        args.labels_field,
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
        "written_records": 0,
        "written_selected_sentences": 0,
        "deferred_records": 0,
        "deferred_selected_sentences": 0,
        "invalid_response_records": 0,
        "input_validation_failed_records": 0,
        "post_check_failed_records": 0,
        "skipped_existing": 0,
        "skipped_by_output": skipped_by_output,
        "input_tokens": 0,
        "output_tokens": 0,
    }
    chunk_size = max(1, args.concurrent_requests * 2)

    async def process_chunk(humanizer: SentenceHumanizer | None) -> None:
        for start in range(0, len(pending_records), chunk_size):
            chunk = pending_records[start:start + chunk_size]
            results = await asyncio.gather(
                *[
                    process_record(
                        index=index,
                        record=record,
                        input_field=args.input_field,
                        output_field=args.output_field,
                        labels_field=args.labels_field,
                        model=args.model,
                        language=args.language,
                        humanizer=humanizer,
                        only_missing_output=args.only_missing_output,
                        dry_run=args.dry_run,
                        fail_on_count_mismatch=args.fail_on_count_mismatch,
                    )
                    for index, record in chunk
                ]
            )
            chunk_success_records: list[dict[str, Any]] = []
            chunk_failure_records: list[dict[str, Any]] = []
            for output_record, failure_record, item_stats in results:
                if output_record is not None:
                    chunk_success_records.append(output_record)
                if failure_record is not None:
                    chunk_failure_records.append(failure_record)
                for key, value in item_stats.items():
                    stats[key] += value
            append_jsonl(output_path, chunk_success_records)
            append_jsonl(failure_log_path, chunk_failure_records)
            logger.info(
                "进度: %d/%d | 本批成功写入 %d 条 | 本批待重跑 %d 条",
                start + len(chunk),
                len(pending_records),
                len(chunk_success_records),
                len(chunk_failure_records),
            )

    if args.dry_run:
        await process_chunk(None)
    else:
        api_logger = ApiCallLogger(
            output_path.parent / "api_logs",
            run_name=f"sentence_humanizer_{args.model}_{input_path.stem}",
        )
        async with SentenceHumanizer(cfg, api_logger=api_logger) as humanizer:
            await process_chunk(humanizer)

    logger.info(
        "完成: written_records=%d written_selected_sentences=%d deferred_records=%d deferred_selected_sentences=%d invalid_response_records=%d input_validation_failed_records=%d post_check_failed_records=%d skipped_existing=%d skipped_by_output=%d input_tokens=%d output_tokens=%d -> %s | failures -> %s",
        stats["written_records"],
        stats["written_selected_sentences"],
        stats["deferred_records"],
        stats["deferred_selected_sentences"],
        stats["invalid_response_records"],
        stats["input_validation_failed_records"],
        stats["post_check_failed_records"],
        stats["skipped_existing"],
        stats["skipped_by_output"],
        stats["input_tokens"],
        stats["output_tokens"],
        output_path,
        failure_log_path,
    )


if __name__ == "__main__":
    asyncio.run(run())
