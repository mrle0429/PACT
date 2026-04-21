"""
Pipeline 主调度模块：将所有步骤串联为完整的数据集构建流程。

流程概览：
    Step 1: 加载 & 过滤人类文本
    Step 2: 为每篇文档分配一个混合模式，并生成各 ratio 变体的 SentenceSelection
    Step 3: 并发调用 LLM 进行 Diff-based 改写
    Step 4: 计算精确连续标签
    Step 5: 写出到 JSONL

断点续传：
    最终输出 JSONL 是唯一完成态真相来源；
    checkpoint 仅保存运行状态与统计信息。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

from .config import DatasetConfig
from .data_loader import SourceDocument, load_human_texts
from .dataset_writer import (
    CheckpointManager,
    DatasetRecord,
    JsonlWriter,
    load_existing_record_ids,
    make_record_id,
)
from .label_calculator import compute_labels
from .llm_rewriter import BaseLLMRewriter, RewriteResult, create_rewriter
from .sentence_processor import (
    SentenceSelection,
    create_sentence_selection,
    enumerate_variants,
    split_into_sentences,
)
from .utils import (
    NonRetryableAPIError,
    RetryExhaustedAPIError,
    get_logger,
)

logger = get_logger(__name__)
TaskResult = tuple[DatasetRecord, int, int] | None


# ---------------------------------------------------------------------------
# 单条任务的数据容器
# ---------------------------------------------------------------------------

@dataclass
class VariantTask:
    """一个待处理的（文档, 变体）任务单元。"""
    doc: SourceDocument
    selection: SentenceSelection
    record_id: str


# ---------------------------------------------------------------------------
# Pipeline 主类
# ---------------------------------------------------------------------------

class DatasetPipeline:
    """
    端到端数据集构建 Pipeline。

    使用方式：
        cfg = DatasetConfig(rewrite_model="gpt-4o-mini", ...)
        pipeline = DatasetPipeline(cfg)
        asyncio.run(pipeline.run())
    """

    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg
        # mixed_dataset.jsonl -> mixed_dataset_gpt-4o-mini.jsonl
        stem = Path(cfg.output_filename).stem
        suffix = Path(cfg.output_filename).suffix
        self._output_path = Path(cfg.output_dir) / f"{stem}_{cfg.rewrite_model}{suffix}"
        self._run_name = _build_run_name(cfg)
        self._checkpoint = CheckpointManager(cfg.checkpoint_dir, self._run_name)
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    # ------------------------------------------------------------------
    # 公共入口
    # ------------------------------------------------------------------
    async def run(
        self,
        max_docs: int | None = None,
        dry_run: bool = False,
    ) -> None:
        """
        执行完整 Pipeline。

        Args:
            max_docs:  仅处理前 N 篇文档（调试用）
            dry_run:   True 时不调用 LLM API，直接用原句填充（测试流程用）
        """
        existing_ids = load_existing_record_ids(self._output_path)
        self._log_run_header(len(existing_ids))

        # Step 1: 加载预采样人类文本
        documents = self._load_documents(max_docs)

        # Step 2: 生成所有变体任务
        all_tasks, pending = self._prepare_pending_tasks(documents, existing_ids)
        self._log_task_overview(all_tasks, pending)

        if not pending:
            logger.info("所有任务已完成，无需重新生成。")
            self._print_summary(len(all_tasks))
            return

        # Step 3-5: 并发处理
        rewriter = None if dry_run else create_rewriter(self.cfg)

        try:
            with JsonlWriter(self._output_path, existing_ids=existing_ids) as writer:
                await self._process_all(pending, rewriter, writer, dry_run=dry_run)
        finally:
            if rewriter is not None:
                await rewriter.aclose()

        self._print_summary(len(all_tasks))


    def _log_run_header(self, restored_count: int) -> None:
        logger.info("=" * 60)
        logger.info("数据集构建 Pipeline 启动")
        logger.info(f"  模型        : {self.cfg.rewrite_model}")
        logger.info(f"  AI 浓度档位 : {self.cfg.ai_ratios}")
        logger.info(f"  混合模式    : {self.cfg.mixing_modes}")
        logger.info(f"  断点恢复    : 输出文件中已有 {restored_count} 条")
        logger.info(f"  输出路径    : {self._output_path}")
        logger.info("=" * 60)

    def _load_documents(self, max_docs: int | None) -> list[SourceDocument]:
        return load_human_texts(
            self.cfg.source_path,
            shuffle=True,
            max_count=max_docs,
            seed=self.cfg.random_seed,
        )

    def _prepare_pending_tasks(
        self,
        documents: list[SourceDocument],
        existing_ids: set[str],
    ) -> tuple[list[VariantTask], list[VariantTask]]:
        all_tasks = self._build_tasks(documents)
        self._sync_runtime_state(all_tasks, existing_ids)
        pending = [task for task in all_tasks if task.record_id not in existing_ids]
        return all_tasks, pending

    def _log_task_overview(
        self,
        all_tasks: list[VariantTask],
        pending: list[VariantTask],
    ) -> None:
        logger.info(
            f"任务概况: 总变体 {len(all_tasks)}, "
            f"已完成 {len(all_tasks) - len(pending)}, "
            f"待处理 {len(pending)}"
        )

   
    # ------------------------------------------------------------------
    # 运行状态同步
    # ------------------------------------------------------------------

    def _sync_runtime_state(
        self,
        all_tasks: list[VariantTask],
        existing_ids: set[str],
    ) -> None:
        """
        启动时用输出文件快照刷新运行状态。
        """
        source_doc_ids = sorted({
            task.doc.doc_id
            for task in all_tasks
            if task.record_id in existing_ids
        })
        changed = self._checkpoint.sync_output_snapshot(
            total_written=len(existing_ids),
            source_doc_ids=source_doc_ids,
        )
        if changed:
            logger.info(
                "运行状态已按输出文件刷新: records=%s, source_docs=%s",
                len(existing_ids),
                len(source_doc_ids),
            )

    # ------------------------------------------------------------------
    # 任务生成
    # ------------------------------------------------------------------

    def _build_doc_tasks(self, doc: SourceDocument, assigned_mode: str) -> list[VariantTask]:
        doc_rng = _build_doc_rng(self.cfg.random_seed, doc.doc_id)
        variants = enumerate_variants(
            doc.text,
            self.cfg.ai_ratios,
            [assigned_mode],
            doc_rng,
        )
        return [
            VariantTask(
                doc=doc,
                selection=selection,
                record_id=make_record_id(doc.doc_id, selection.target_ratio, selection.mode),
            )
            for selection in variants
        ]

    def _build_tasks(self, documents: list[SourceDocument]) -> list[VariantTask]:
        """
        为每篇文档生成所有 ratio 变体的 VariantTask。

        每篇文档只分配一种混合模式（block_replace 或 random_scatter），
        两种模式在文档级别保持约 1:1（奇数篇时数量差最多 1）。

        重要：模式分配不依赖 max_docs 总量，保证断点续跑时同一文档的
        record_id（含 mode 后缀）稳定可复用。
        """
        tasks: list[VariantTask] = []
        modes = _deduped_mixing_modes(self.cfg.mixing_modes)

        # 优先使用源文档中固化的 fixed_mixing_mode，避免因 source 集合变化或 shuffle
        # 变化导致同一 doc_id 的 mode 漂移。旧数据若缺少该字段，则回退到历史分配。
        for doc_idx, doc in enumerate(documents):
            assigned_mode = _resolve_doc_mode(doc, doc_idx, modes, self.cfg.random_seed)
            tasks.extend(self._build_doc_tasks(doc, assigned_mode))
        return tasks

    # ------------------------------------------------------------------
    # 并发处理主循环
    # ------------------------------------------------------------------

    def _write_completed_record(
        self,
        writer: JsonlWriter,
        task: VariantTask,
        record: DatasetRecord,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        if writer.write(record):
            self._checkpoint.record_write(
                source_doc_id=task.doc.doc_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens

    async def _process_all(
        self,
        tasks: list[VariantTask],
        rewriter: BaseLLMRewriter | None,
        writer: JsonlWriter,
        dry_run: bool,
    ) -> None:
        """
        以 asyncio.gather + Semaphore 并发执行所有 VariantTask。
        实时打印进度条（含 ETA 与 token 统计）。
        """
        semaphore = asyncio.Semaphore(self.cfg.concurrent_requests)
        processed_count = 0
        success_count = 0
        skipped_count = 0
        total = len(tasks)
        start_ts = time.monotonic()
        last_render_ts = 0.0

        # 打印进度
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

        # 内部协程。执行单个任务
        async def _process_one(task: VariantTask) -> None:
            nonlocal processed_count, success_count, skipped_count
            async with semaphore:
                result = await self._process_task(task, rewriter, dry_run)
            if result is not None:
                record, in_tok, out_tok = result
                self._write_completed_record(writer, task, record, in_tok, out_tok)
                success_count += 1
            else:
                skipped_count += 1
            processed_count += 1
            _render_progress(force=(processed_count == total))

        await asyncio.gather(*(_process_one(task) for task in tasks))

    # ------------------------------------------------------------------
    # 单任务处理
    # ------------------------------------------------------------------

    async def _resolve_rewrites(
        self,
        task: VariantTask,
        rewriter: BaseLLMRewriter | None,
        dry_run: bool,
    ) -> RewriteResult:
        
        sel = task.selection

        if rewriter is None:
            raise ValueError("rewriter 不能为空（dry_run=False 时必须提供）")

        if sel.target_ratio == 0.0 or not sel.selected_indices:
            model_id = rewriter.model_cfg.model_id if rewriter is not None else self.cfg.rewrite_model
            return RewriteResult({}, model_id)

        if dry_run:
            rewrites = {i: sel.sentences[i] for i in sel.selected_indices}
            return RewriteResult(rewrites, "dry_run")


        return await rewriter.rewrite(
            sel.sentences,
            sel.selected_indices,
            task_id=task.record_id,
        )

    def _is_incomplete_rewrite(
        self,
        selection: SentenceSelection,
        result: RewriteResult,
        dry_run: bool,
    ) -> bool:
        if dry_run or selection.target_ratio == 0.0 or not selection.selected_indices:
            return False
        if result.missing_indices or result.invalid_indices or result.extra_indices:
            return True
        return len(result.rewrites) < len(selection.selected_indices)

    def _build_record(
        self,
        task: VariantTask,
        rewrites: dict[int, str],
        dry_run: bool,
    ) -> DatasetRecord:
        sel = task.selection
        doc = task.doc
        mixed_sentences = sel.build_mixed_sentences(rewrites)
        label_dict = compute_labels(
            original_text=doc.text,
            mixed_sentences=mixed_sentences,
            ai_indices=list(rewrites.keys()),
            cfg=self.cfg,
        )

        return DatasetRecord(
            id=task.record_id,
            source_dataset=doc.source_dataset,
            source_domain=doc.domain,
            original_text=doc.text,
            mixed_text=" ".join(mixed_sentences),
            n_sentences=sel.n,
            target_ai_ratio=sel.target_ratio,
            mixing_mode=sel.mode,
            rewrite_model=(
                self.cfg.rewrite_model
                if (not dry_run and sel.target_ratio > 0.0)
                else ("human" if sel.target_ratio == 0.0 else "dry_run")
            ),
            sentence_labels=sel.sentence_label_array(rewrites),
            lir=label_dict.get("lir", 0.0),
            jaccard_distance=label_dict.get("jaccard_distance"),
            sentence_jaccard=label_dict.get("sentence_jaccard"),
            cosine_distance=label_dict.get("cosine_distance"),
        )

    async def _process_task(
        self,
        task: VariantTask,
        rewriter: BaseLLMRewriter | None,
        dry_run: bool,
    ) -> TaskResult:
        """处理单个 VariantTask，返回 (DatasetRecord, in_tok, out_tok) 或 None（跳过）。"""
        try:
            result = await self._resolve_rewrites(task, rewriter, dry_run)
            rewrites = result.rewrites
            in_tok, out_tok = result.input_tokens, result.output_tokens

            if self._is_incomplete_rewrite(task.selection, result, dry_run):
                expected = len(task.selection.selected_indices)
                logger.warning(
                    f"[{task.record_id}] 改写结果非法（expected={expected}, got={len(rewrites)}; "
                    f"missing={len(result.missing_indices)}, invalid={len(result.invalid_indices)}, extra={len(result.extra_indices)}），"
                    "该样本已跳过，不写入数据集。"
                )
                return None

            final_sentence_count = _count_mixed_sentence_count(task.selection, rewrites)
            if final_sentence_count != task.selection.n:
                logger.warning(
                    f"[{task.record_id}] 回填后句子数发生变化（before={task.selection.n}, "
                    f"after={final_sentence_count}），"
                    "该样本已跳过，不写入数据集。"
                )
                return None

            record = self._build_record(task, rewrites, dry_run)
            return record, in_tok, out_tok

        except NonRetryableAPIError as exc:
            if _is_content_inspection_error(exc):
                logger.warning(
                    f"[{task.record_id}] 命中平台内容审查，已跳过该样本: {type(exc).__name__}: {exc}"
                )
                return None
            logger.error(
                f"[{task.record_id}] API 错误不可恢复，程序终止: "
                f"{type(exc).__name__}: {exc}"
            )
            raise
        except RetryExhaustedAPIError as exc:
            logger.error(
                f"[{task.record_id}] API 重试耗尽，程序终止: "
                f"{type(exc).__name__}: {exc}"
            )
            raise
        except Exception as exc:
            logger.error(f"[{task.record_id}] 处理失败，已跳过: {type(exc).__name__}: {exc}")
            return None

    # ------------------------------------------------------------------
    # 摘要
    # ------------------------------------------------------------------

    def _print_summary(self, total_tasks: int) -> None:
        stats = self._checkpoint.stats
        logger.info("=" * 60)
        logger.info("Pipeline 完成")
        logger.info(f"  总变体数        : {total_tasks}")
        logger.info(f"  成功写入        : {stats.get('total_written', 0)}")
        logger.info(f"  原始文本数      : {stats.get('total_source_docs_processed', 0)}")
        logger.info(f"  API 输入 Tokens : {stats.get('api_input_tokens', 0):,}")
        logger.info(f"  API 输出 Tokens : {stats.get('api_output_tokens', 0):,}")
        logger.info(f"  输出文件        : {self._output_path}")
        logger.info("=" * 60)


# ---------------------------------------------------------------------------
# 单文本处理接口（供交互式测试 / 外部调用）
# ---------------------------------------------------------------------------

@dataclass
class SingleTextResult:
    """单文本处理的返回结果。"""
    original_text: str
    mixed_text: str
    sentences: list[str]
    selected_indices: list[int]
    rewrites: dict[int, str]
    labels: dict
    sentence_labels: list[int]
    model_id: str
    mixing_mode: str
    target_ratio: float
    input_tokens: int = 0
    output_tokens: int = 0

    def summary(self) -> str:
        """返回易读的文字摘要。"""
        lines = [
            f"{'='*60}",
            f"模型: {self.model_id}  |  模式: {self.mixing_mode}  |  目标浓度: {self.target_ratio:.0%}",
            f"句子总数: {len(self.sentences)}  |  选中改写: {len(self.selected_indices)}",
            f"{'='*60}",
            "",
            "【原文】",
            self.original_text,
            "",
            "【混合文本】",
            self.mixed_text,
            "",
            "【句子级标签】",
            "  ".join(f"[{i}]{'AI' if l else 'HU'}" for i, l in enumerate(self.sentence_labels)),
            "",
            "【标签值】",
            f"  LIR:              {self.labels.get('lir', 'N/A')}",
            f"  Jaccard Distance: {self.labels.get('jaccard_distance', 'N/A')}",
            f"  Sentence Jaccard: {self.labels.get('sentence_jaccard', 'N/A')}",
            f"  Cosine Distance:  {self.labels.get('cosine_distance', 'N/A')}",
            "",
            f"API Tokens: in={self.input_tokens}, out={self.output_tokens}",
        ]
        return "\n".join(lines)

    def to_dataset_record(
        self,
        record_id: str = "",
        source_dataset: str = "single_test",
        domain: str = "test",
    ) -> DatasetRecord:
        """转换为与批量 Pipeline 完全一致的 DatasetRecord 格式。"""
        if not record_id:
            record_id = _make_single_text_record_id(
                self.original_text,
                self.target_ratio,
                self.mixing_mode,
            )

        return DatasetRecord(
            id=record_id,
            source_dataset=source_dataset,
            source_domain=domain,
            original_text=self.original_text,
            mixed_text=self.mixed_text,
            n_sentences=len(self.sentences),
            target_ai_ratio=self.target_ratio,
            mixing_mode=self.mixing_mode,
            rewrite_model=self.model_id,
            sentence_labels=self.sentence_labels,
            lir=self.labels.get("lir", 0.0),
            jaccard_distance=self.labels.get("jaccard_distance"),
            sentence_jaccard=self.labels.get("sentence_jaccard"),
            cosine_distance=self.labels.get("cosine_distance"),
        )


async def process_single_text(
    text: str,
    target_ratio: float = 0.4,
    mixing_mode: str = "block_replace",
    model: str = "MiniMax-M2.7",
    seed: int = 42,
    dry_run: bool = False,
    language_hint: str = "English",
) -> SingleTextResult:
    """
    对单条人类文本进行完整的混合构造与标签计算。

    这是最细粒度的接口，适用于：
    - 交互式调试和快速实验
    - 从 Notebook 中调用
    - 集成到其他系统中

    Args:
        text:          输入的人类原文
        target_ratio:  AI 浓度 ∈ [0.0, 1.0]
        mixing_mode:   "block_replace" 或 "random_scatter"
        model:         模型名称（见 SUPPORTED_MODELS）
        seed:          随机种子
        dry_run:       True 时不调用 API，用原句占位
        language_hint: 语言提示（传给改写 Prompt）

    Returns:
        SingleTextResult — 包含混合文本、标签、改写详情
    """
    cfg = DatasetConfig(rewrite_model=model, random_seed=seed)
    rng = random.Random(seed)

    # Step 1: 分句
    sentences = split_into_sentences(text)
    if not sentences:
        raise ValueError("输入文本分句后为空，请检查文本内容。")

    # Step 2: 选取句子
    selection = create_sentence_selection(sentences, target_ratio, mixing_mode, rng)  # type: ignore[arg-type]

    # Step 3: 改写
    in_tok, out_tok = 0, 0
    if not selection.selected_indices or target_ratio == 0.0:
        rewrites: dict[int, str] = {}
    elif dry_run:
        rewrites = {i: sentences[i] for i in selection.selected_indices}
    else:
        rewriter = create_rewriter(cfg)
        try:
            result = await rewriter.rewrite(sentences, selection.selected_indices, language_hint)
            if result.missing_indices or result.invalid_indices or result.extra_indices:
                raise ValueError(
                    "改写结果非法: "
                    f"missing={result.missing_indices}, "
                    f"invalid={result.invalid_indices}, "
                    f"extra={result.extra_indices}, "
                    f"error={result.error}"
                )
            rewrites = result.rewrites
            in_tok, out_tok = result.input_tokens, result.output_tokens
        finally:
            await rewriter.aclose()

    # Step 4: 回填
    mixed_sentences = selection.build_mixed_sentences(rewrites)
    mixed_text = " ".join(mixed_sentences)
    final_sentence_count = len(split_into_sentences(mixed_text))
    if final_sentence_count != len(sentences):
        raise ValueError(
            f"回填后句子数发生变化: before={len(sentences)}, after={final_sentence_count}"
        )
    sentence_labels = selection.sentence_label_array(rewrites)

    # Step 5: 标签计算
    labels = compute_labels(
        original_text=text,
        mixed_sentences=mixed_sentences,
        ai_indices=list(rewrites.keys()),
        cfg=cfg,
    )

    return SingleTextResult(
        original_text=text,
        mixed_text=mixed_text,
        sentences=sentences,
        selected_indices=selection.selected_indices,
        rewrites=rewrites,
        labels=labels,
        sentence_labels=sentence_labels,
        model_id=model if not dry_run else "dry_run",
        mixing_mode=mixing_mode,
        target_ratio=target_ratio,
        input_tokens=in_tok,
        output_tokens=out_tok,
    )


def process_single_text_sync(
    text: str,
    target_ratio: float = 0.4,
    mixing_mode: str = "block_replace",
    model: str = "MiniMax-M2.7",
    seed: int = 42,
    dry_run: bool = False,
    language_hint: str = "English",
) -> SingleTextResult:
    """
    process_single_text 的同步包装，方便在非异步环境中直接调用。

    用法：
        from src.pipeline import process_single_text_sync

        result = process_single_text_sync(
            "The sky is blue. Water is wet. ...",
            target_ratio=0.4,
            mixing_mode="block_replace",
            model="MiniMax-M2.7",
            dry_run=True,
        )
        print(result.summary())
    """
    return asyncio.run(process_single_text(
        text=text,
        target_ratio=target_ratio,
        mixing_mode=mixing_mode,
        model=model,
        seed=seed,
        dry_run=dry_run,
        language_hint=language_hint,
    ))


# ---------------------------------------------------------------------------
# 文件尾部辅助函数
# ---------------------------------------------------------------------------

def _build_run_name(cfg: DatasetConfig) -> str:
    """
    生成带配置指纹的 run_name，避免不同配置复用同一个 checkpoint。
    """
    signature_payload = {
        "source_path": cfg.source_path,
        "source_tag": cfg.source_tag,
        "ai_ratios": cfg.ai_ratios,
        "mixing_modes": cfg.mixing_modes,
        "rewrite_model": cfg.rewrite_model,
        "random_seed": cfg.random_seed,
        "tokenizer_for_lir": cfg.tokenizer_for_lir,
        "ngram_n": cfg.ngram_n,
    }
    payload = json.dumps(signature_payload, ensure_ascii=True, sort_keys=True)
    signature = hashlib.md5(payload.encode("utf-8")).hexdigest()[:10]
    return f"run_{cfg.rewrite_model}_{cfg.source_tag}_{signature}"


def _deduped_mixing_modes(mixing_modes: list[str]) -> list[str]:
    modes = list(dict.fromkeys(mixing_modes))
    return modes or ["block_replace", "random_scatter"]


def _resolve_doc_mode(
    doc: SourceDocument,
    doc_idx: int,
    modes: list[str],
    random_seed: int,
) -> str:
    fixed_mode = doc.fixed_mixing_mode
    if fixed_mode:
        if fixed_mode in modes:
            return fixed_mode
        logger.warning(
            f"[{doc.doc_id}] fixed_mixing_mode={fixed_mode!r} 不在当前配置 {modes} 中，"
            "已回退到旧的稳定分配逻辑。"
        )
    return _assigned_mode_for_doc(doc_idx, modes, random_seed)


def _assigned_mode_for_doc(doc_idx: int, modes: list[str], random_seed: int) -> str:
    mode_offset = random_seed % len(modes)
    return modes[(doc_idx + mode_offset) % len(modes)]


def _build_doc_rng(random_seed: int, doc_id: str) -> random.Random:
    doc_seed = int(
        hashlib.md5(f"{random_seed}:{doc_id}".encode("utf-8")).hexdigest()[:8],
        16,
    )
    return random.Random(doc_seed)


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


def _count_mixed_sentence_count(
    selection: SentenceSelection,
    rewrites: dict[int, str],
) -> int:
    return len(split_into_sentences(selection.build_mixed_text(rewrites)))


def _is_content_inspection_error(exc: NonRetryableAPIError) -> bool:
    error_text = str(exc).lower()
    return (
        "data_inspection_failed" in error_text
        or "inappropriate content" in error_text
    )


def _make_single_text_record_id(
    original_text: str,
    target_ratio: float,
    mixing_mode: str,
) -> str:
    text_hash = hashlib.md5(original_text.encode()).hexdigest()[:8]
    return make_record_id(f"single_{text_hash}", target_ratio, mixing_mode)
