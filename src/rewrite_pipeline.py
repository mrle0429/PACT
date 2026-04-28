"""
二次 rewrite 数据集构建 pipeline。

输入：
- `mixed_dataset_*.jsonl`

输出：
- `rewrite_*.jsonl`

核心规则：
- 仅对 `sentence_labels == 1` 的句子做二次 humanize
- 非目标句保持 `mixed_text` 原样
- 严格校验返回 JSON、句子数与单句边界
- 不合法样本直接跳过，不写入最终数据集
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import DatasetConfig
from .dataset_writer import CheckpointManager, JsonlWriter, load_existing_record_ids
from .humanizer import SentenceHumanizer
from .sentence_processor import split_into_sentences
from .utils import NonRetryableAPIError, RetryExhaustedAPIError, get_logger

logger = get_logger(__name__)
TaskResult = tuple["SentenceRewriteRecord", int, int] | None


@dataclass
class RewriteSourceRecord:
    line_index: int
    id: str
    source_dataset: str
    source_domain: str
    original_text: str
    mixed_text: str
    n_sentences: int
    target_ai_ratio: float
    sentence_labels: list[int]
    mixing_mode: str = ""

    @property
    def selected_indices(self) -> list[int]:
        return [index for index, label in enumerate(self.sentence_labels) if label == 1]


@dataclass
class SentenceRewriteRecord:
    id: str
    source_dataset: str
    source_model: str
    source_domain: str
    original_text: str
    mixed_text: str
    rewritten_text: str
    target_ai_ratio: float
    n_sentences: int
    sentence_labels: list[int]
    ai_sentences_original: list[str] = field(default_factory=list)
    ai_sentences_rewritten: list[str] = field(default_factory=list)
    mixing_mode: str = ""
    rewrite_info: dict[str, Any] = field(default_factory=dict)

    def to_jsonl_line(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass
class RewriteTask:
    source: RewriteSourceRecord

    @property
    def record_id(self) -> str:
        return self.source.id


def normalize_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return str(value)


def normalize_labels(value: Any, task_id: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"[{task_id}] sentence_labels 必须是 list，实际为 {type(value).__name__}")

    normalized: list[int] = []
    for index, item in enumerate(value):
        try:
            label = int(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"[{task_id}] sentence_labels[{index}] 无法转换为整数: {item!r}"
            ) from exc
        if label not in {0, 1}:
            raise ValueError(f"[{task_id}] sentence_labels[{index}] 必须为 0/1，实际为 {label}")
        normalized.append(label)
    return normalized


def load_rewrite_source_records(
    input_path: str | Path,
    *,
    max_records: int | None = None,
) -> list[RewriteSourceRecord]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"输入 mixed_dataset 文件不存在: {path.resolve()}")

    records: list[RewriteSourceRecord] = []
    with path.open(encoding="utf-8") as file_obj:
        for line_no, raw_line in enumerate(file_obj, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("输入文件第 %d 行 JSON 解析失败，已跳过: %s", line_no, exc)
                continue
            if not isinstance(payload, dict):
                logger.warning("输入文件第 %d 行不是对象，已跳过。", line_no)
                continue

            task_id = str(payload.get("id", f"line_{line_no}"))
            try:
                labels = normalize_labels(payload.get("sentence_labels", []), task_id)
            except ValueError as exc:
                logger.warning("%s 已跳过。", exc)
                continue

            try:
                target_ai_ratio = float(payload.get("target_ai_ratio", 0.0))
            except (TypeError, ValueError):
                target_ai_ratio = 0.0

            try:
                n_sentences = int(payload.get("n_sentences", len(labels)))
            except (TypeError, ValueError):
                n_sentences = len(labels)

            records.append(
                RewriteSourceRecord(
                    line_index=line_no - 1,
                    id=task_id,
                    source_dataset=normalize_text(payload.get("source_dataset")),
                    source_domain=normalize_text(payload.get("source_domain")),
                    original_text=normalize_text(payload.get("original_text")),
                    mixed_text=normalize_text(payload.get("mixed_text")),
                    n_sentences=n_sentences,
                    target_ai_ratio=target_ai_ratio,
                    sentence_labels=labels,
                    mixing_mode=normalize_text(payload.get("mixing_mode")),
                )
            )

            if max_records is not None and len(records) >= max_records:
                break

    logger.info("mixed_dataset 加载完成: %d 条 (来源: %s)", len(records), path.name)
    return records


def infer_source_model(input_path: str | Path) -> str | None:
    path = Path(input_path)
    match = re.match(r"^mixed_dataset_(.+)\.jsonl$", path.name)
    if match:
        return match.group(1)

    candidates: set[str] = set()
    with path.open(encoding="utf-8") as file_obj:
        for raw_line in file_obj:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            value = normalize_text(payload.get("rewrite_model")).strip()
            if value and value not in {"human", "dry_run"}:
                candidates.add(value)
            if len(candidates) > 1:
                return None
    if len(candidates) == 1:
        return next(iter(candidates))
    return None


def build_default_output_path(
    input_path: str | Path,
    source_model: str,
    rewriter_model: str,
) -> Path:
    base_dir = Path(input_path).resolve().parent
    if source_model == rewriter_model:
        filename = f"rewrite_{source_model}.jsonl"
    else:
        filename = f"rewrite_{source_model}__by_{rewriter_model}.jsonl"
    return base_dir / filename


class RewriteDatasetPipeline:
    def __init__(
        self,
        cfg: DatasetConfig,
        *,
        input_path: str | Path,
        output_path: str | Path,
        source_model: str,
        language_hint: str = "English",
    ):
        self.cfg = cfg
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.source_model = source_model
        self.language_hint = language_hint
        self._run_name = _build_run_name(
            cfg=cfg,
            input_path=self.input_path,
            output_path=self.output_path,
            source_model=source_model,
            language_hint=language_hint,
        )
        self._checkpoint = CheckpointManager(cfg.checkpoint_dir, self._run_name)
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._run_date = datetime.now().date().isoformat()

    async def run(
        self,
        *,
        max_records: int | None = None,
        dry_run: bool = False,
    ) -> None:
        existing_ids = load_existing_record_ids(self.output_path)
        self._log_run_header(len(existing_ids), dry_run)

        source_records = load_rewrite_source_records(
            self.input_path,
            max_records=max_records,
        )
        all_tasks = [RewriteTask(source=record) for record in source_records]
        self._sync_runtime_state(all_tasks, existing_ids)
        pending = [task for task in all_tasks if task.record_id not in existing_ids]

        logger.info(
            "任务概况: 总记录 %d, 已完成 %d, 待处理 %d",
            len(all_tasks),
            len(all_tasks) - len(pending),
            len(pending),
        )

        if not pending:
            logger.info("所有任务已完成，无需重新生成。")
            self._print_summary(len(all_tasks))
            return

        humanizer = None
        if not dry_run:
            api_logger = _build_api_logger(self.output_path, self.source_model, self.cfg.rewrite_model)
            humanizer = SentenceHumanizer(self.cfg, api_logger=api_logger)

        try:
            with JsonlWriter(self.output_path, existing_ids=existing_ids) as writer:
                await self._process_all(pending, humanizer, writer, dry_run=dry_run)
        finally:
            if humanizer is not None:
                await humanizer.aclose()

        self._print_summary(len(all_tasks))

    def _log_run_header(self, restored_count: int, dry_run: bool) -> None:
        logger.info("=" * 60)
        logger.info("二次 rewrite Pipeline 启动")
        logger.info("  输入文件      : %s", self.input_path)
        logger.info("  输出文件      : %s", self.output_path)
        logger.info("  source_model  : %s", self.source_model)
        logger.info("  rewriter      : %s", self.cfg.rewrite_model)
        logger.info("  并发数        : %s", self.cfg.concurrent_requests)
        logger.info("  断点恢复      : 输出文件中已有 %s 条", restored_count)
        logger.info("  dry_run       : %s", dry_run)
        logger.info("=" * 60)

    def _sync_runtime_state(
        self,
        all_tasks: list[RewriteTask],
        existing_ids: set[str],
    ) -> None:
        source_record_ids = sorted(
            task.record_id
            for task in all_tasks
            if task.record_id in existing_ids
        )
        changed = self._checkpoint.sync_output_snapshot(
            total_written=len(existing_ids),
            source_doc_ids=source_record_ids,
        )
        if changed:
            logger.info(
                "运行状态已按输出文件刷新: records=%s",
                len(existing_ids),
            )

    def _write_completed_record(
        self,
        writer: JsonlWriter,
        task: RewriteTask,
        record: SentenceRewriteRecord,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        if writer.write(record):
            self._checkpoint.record_write(
                source_doc_id=task.record_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens

    async def _process_all(
        self,
        tasks: list[RewriteTask],
        humanizer: SentenceHumanizer | None,
        writer: JsonlWriter,
        *,
        dry_run: bool,
    ) -> None:
        semaphore = asyncio.Semaphore(self.cfg.concurrent_requests)
        processed_count = 0
        success_count = 0
        skipped_count = 0
        total = len(tasks)
        start_ts = time.monotonic()
        last_render_ts = 0.0

        def _render_progress(force: bool = False) -> None:
            nonlocal last_render_ts
            now = time.monotonic()
            if (not force) and (now - last_render_ts < 1.0):
                return

            elapsed = max(0.001, now - start_ts)
            speed = processed_count / elapsed
            remaining = max(0, total - processed_count)
            eta = int(remaining / speed) if speed > 0 else -1
            logger.info(
                _build_progress_message(
                    processed_count=processed_count,
                    total=total,
                    success_count=success_count,
                    skipped_count=skipped_count,
                    speed=speed,
                    eta=eta,
                    input_tokens=self._total_input_tokens,
                    output_tokens=self._total_output_tokens,
                )
            )
            last_render_ts = now

        async def _process_one(task: RewriteTask) -> None:
            nonlocal processed_count, success_count, skipped_count
            async with semaphore:
                result = await self._process_task(task, humanizer, dry_run=dry_run)
            if result is not None:
                record, in_tok, out_tok = result
                self._write_completed_record(writer, task, record, in_tok, out_tok)
                success_count += 1
            else:
                skipped_count += 1
            processed_count += 1
            _render_progress(force=(processed_count == total))

        await asyncio.gather(*(_process_one(task) for task in tasks))

    async def _process_task(
        self,
        task: RewriteTask,
        humanizer: SentenceHumanizer | None,
        *,
        dry_run: bool,
    ) -> TaskResult:
        source = task.source
        try:
            if not source.mixed_text.strip():
                logger.warning("[%s] mixed_text 为空，已跳过。", source.id)
                return None

            sentences = split_into_sentences(source.mixed_text)
            if len(sentences) != len(source.sentence_labels):
                logger.warning(
                    "[%s] mixed_text 分句数与 sentence_labels 不一致 (sentences=%d, labels=%d)，已跳过。",
                    source.id,
                    len(sentences),
                    len(source.sentence_labels),
                )
                return None
            if source.n_sentences != len(source.sentence_labels):
                logger.warning(
                    "[%s] n_sentences 与 sentence_labels 长度不一致 (n_sentences=%d, labels=%d)，已跳过。",
                    source.id,
                    source.n_sentences,
                    len(source.sentence_labels),
                )
                return None

            selected_indices = source.selected_indices
            if not dry_run and selected_indices and humanizer is None:
                raise ValueError("humanizer 不能为空（dry_run=False 时必须提供）")

            if not selected_indices:
                result = None
                rewrites: dict[int, str] = {}
                input_tokens = 0
                output_tokens = 0
                prompt = ""
            elif dry_run:
                result = None
                rewrites = {index: sentences[index] for index in selected_indices}
                input_tokens = 0
                output_tokens = 0
                prompt = ""
            else:
                result = await humanizer.humanize_selected(
                    sentences=sentences,
                    selected_indices=selected_indices,
                    language_hint=self.language_hint,
                    task_id=source.id,
                )
                rewrites = result.rewrites
                input_tokens = result.input_tokens
                output_tokens = result.output_tokens
                prompt = result.prompt

                if _is_incomplete_rewrite(selected_indices, result):
                    logger.warning(
                        "[%s] 二次 rewrite 结果非法（expected=%d, got=%d; missing=%d, invalid=%d, extra=%d），该样本已跳过。",
                        source.id,
                        len(selected_indices),
                        len(rewrites),
                        len(result.missing_indices),
                        len(result.invalid_indices),
                        len(result.extra_indices),
                    )
                    return None

            final_sentences = list(sentences)
            for index, rewritten in rewrites.items():
                if 0 <= index < len(final_sentences) and rewritten.strip():
                    final_sentences[index] = rewritten.strip()

            rewritten_text = " ".join(final_sentences)
            final_sentence_count = len(split_into_sentences(rewritten_text))
            if final_sentence_count != len(sentences):
                logger.warning(
                    "[%s] 回填后句子数发生变化（before=%d, after=%d），该样本已跳过。",
                    source.id,
                    len(sentences),
                    final_sentence_count,
                )
                return None

            record = self._build_output_record(
                source=source,
                sentences=sentences,
                final_sentences=final_sentences,
                prompt=prompt,
                dry_run=dry_run,
            )
            return record, input_tokens, output_tokens

        except NonRetryableAPIError as exc:
            if _is_content_inspection_error(exc):
                logger.warning(
                    "[%s] 命中平台内容审查，已跳过该样本: %s: %s",
                    source.id,
                    type(exc).__name__,
                    exc,
                )
                return None
            logger.error(
                "[%s] API 错误不可恢复，程序终止: %s: %s",
                source.id,
                type(exc).__name__,
                exc,
            )
            raise
        except RetryExhaustedAPIError as exc:
            logger.error(
                "[%s] API 重试耗尽，程序终止: %s: %s",
                source.id,
                type(exc).__name__,
                exc,
            )
            raise
        except Exception as exc:
            logger.error("[%s] 处理失败，已跳过: %s: %s", source.id, type(exc).__name__, exc)
            return None

    def _build_output_record(
        self,
        *,
        source: RewriteSourceRecord,
        sentences: list[str],
        final_sentences: list[str],
        prompt: str,
        dry_run: bool,
    ) -> SentenceRewriteRecord:
        selected_indices = source.selected_indices
        ai_sentences_original = [sentences[index] for index in selected_indices]
        ai_sentences_rewritten = [final_sentences[index] for index in selected_indices]

        if not selected_indices:
            status = "no_selected_sentences"
        elif dry_run:
            status = "dry_run"
        else:
            status = "ok"

        return SentenceRewriteRecord(
            id=source.id,
            source_dataset=source.source_dataset,
            source_model=self.source_model,
            source_domain=source.source_domain,
            original_text=source.original_text,
            mixed_text=source.mixed_text,
            rewritten_text=" ".join(final_sentences),
            target_ai_ratio=source.target_ai_ratio,
            n_sentences=source.n_sentences,
            sentence_labels=list(source.sentence_labels),
            ai_sentences_original=ai_sentences_original,
            ai_sentences_rewritten=ai_sentences_rewritten,
            mixing_mode=source.mixing_mode,
            rewrite_info={
                "rewriter": self.cfg.rewrite_model if not dry_run else "dry_run",
                "prompt": prompt,
                "timestamp": self._run_date,
                "status": status,
            },
        )

    def _print_summary(self, total_tasks: int) -> None:
        stats = self._checkpoint.stats
        logger.info("=" * 60)
        logger.info("二次 rewrite Pipeline 完成")
        logger.info("  总记录数        : %s", total_tasks)
        logger.info("  成功写入        : %s", stats.get("total_written", 0))
        logger.info("  已完成记录数    : %s", stats.get("total_source_docs_processed", 0))
        logger.info("  API 输入 Tokens : %s", f"{stats.get('api_input_tokens', 0):,}")
        logger.info("  API 输出 Tokens : %s", f"{stats.get('api_output_tokens', 0):,}")
        logger.info("  输出文件        : %s", self.output_path)
        logger.info("=" * 60)


def _build_api_logger(
    output_path: Path,
    source_model: str,
    rewriter_model: str,
):
    from .rewriters import ApiCallLogger

    run_name = f"rewrite_{source_model}__by_{rewriter_model}_{output_path.stem}"
    return ApiCallLogger(output_path.parent / "api_logs", run_name=run_name)


def _is_incomplete_rewrite(
    selected_indices: list[int],
    result,
) -> bool:
    if not selected_indices:
        return False
    if result.missing_indices or result.invalid_indices or result.extra_indices:
        return True
    return len(result.rewrites) < len(selected_indices)


def _build_run_name(
    *,
    cfg: DatasetConfig,
    input_path: Path,
    output_path: Path,
    source_model: str,
    language_hint: str,
) -> str:
    signature_payload = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "source_model": source_model,
        "rewriter_model": cfg.rewrite_model,
        "language_hint": language_hint,
        "random_seed": cfg.random_seed,
    }
    payload = json.dumps(signature_payload, ensure_ascii=True, sort_keys=True)
    signature = hashlib.md5(payload.encode("utf-8")).hexdigest()[:10]
    return f"rewrite_{source_model}_{cfg.rewrite_model}_{signature}"


def _build_progress_message(
    *,
    processed_count: int,
    total: int,
    success_count: int,
    skipped_count: int,
    speed: float,
    eta: int,
    input_tokens: int,
    output_tokens: int,
) -> str:
    ratio = (processed_count / total) if total else 1.0
    width = 28
    done = int(width * ratio)
    bar = f"[{'#' * done}{'-' * (width - done)}]"
    eta_str = f"{eta}s" if eta >= 0 else "N/A"
    return (
        f"进度 {bar} {ratio * 100:6.2f}% "
        f"{processed_count}/{total} | 成功 {success_count} | 跳过 {skipped_count} | "
        f"{speed:.2f} task/s | ETA {eta_str} | "
        f"tokens in={input_tokens}, out={output_tokens}"
    )


def _is_content_inspection_error(exc: NonRetryableAPIError) -> bool:
    error_text = str(exc).lower()
    return (
        "data_inspection_failed" in error_text
        or "internalerror.algo.datainspectionfailed" in error_text
        or "inappropriate content" in error_text
    )
