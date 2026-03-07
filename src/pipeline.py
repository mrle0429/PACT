"""
Pipeline 主调度模块：将所有步骤串联为完整的数据集构建流程。

流程概览：
    Step 1: 加载 & 过滤人类文本
    Step 2: 为每篇文档分配一个混合模式，并生成各 ratio 变体的 SentenceSelection
    Step 3: 并发调用 LLM 进行 Diff-based 改写
    Step 4: 计算精确连续标签
    Step 5: 写出到 JSONL

断点续传：
    CheckpointManager 记录已生成的 record_id，
    重新运行时跳过已完成的，无缝续接。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .config import DatasetConfig
from .data_loader import SourceDocument, load_human_texts
from .dataset_writer import (
    CheckpointManager,
    DatasetRecord,
    JsonlWriter,
    load_existing_record_ids,
    make_record_id,
    record_id_to_variant_key,
)
from .label_calculator import compute_labels
from .llm_rewriter import BaseLLMRewriter, create_rewriter
from .sentence_processor import (
    SentenceSelection,
    enumerate_variants,
)
from .utils import (
    NonRetryableAPIError,
    RetryExhaustedAPIError,
    estimate_cost_breakdown_usd,
    estimate_cost_usd,
    estimate_request_cost_usd,
    get_logger,
)

logger = get_logger(__name__)


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
        self._rng = random.Random(cfg.random_seed)
        # 输出文件名末尾带模型名，如 mixed_dataset_qwen3.5-plus.jsonl
        stem = Path(cfg.output_filename).stem
        suffix = Path(cfg.output_filename).suffix
        self._output_path = Path(cfg.output_dir) / f"{stem}_{cfg.rewrite_model}{suffix}"
        self._run_name = self._build_run_name()
        self._checkpoint = CheckpointManager(cfg.checkpoint_dir, self._run_name)
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._estimated_cost_usd = float(self._checkpoint.stats.get("api_estimated_cost_usd", 0.0))

    def _build_run_name(self) -> str:
        """
        生成带配置指纹的 run_name，避免不同配置复用同一个 checkpoint。
        """
        signature_payload = {
            "source_path": self.cfg.source_path,
            "source_tag": self.cfg.source_tag,
            "ai_ratios": self.cfg.ai_ratios,
            "mixing_modes": self.cfg.mixing_modes,
            "rewrite_model": self.cfg.rewrite_model,
            "random_seed": self.cfg.random_seed,
            "tokenizer_for_lir": self.cfg.tokenizer_for_lir,
            "ngram_n": self.cfg.ngram_n,
        }
        payload = json.dumps(signature_payload, ensure_ascii=True, sort_keys=True)
        signature = hashlib.md5(payload.encode("utf-8")).hexdigest()[:10]
        return f"run_{self.cfg.rewrite_model}_{self.cfg.source_tag}_{signature}"

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
        logger.info("=" * 60)
        logger.info("数据集构建 Pipeline 启动")
        logger.info(f"  模型        : {self.cfg.rewrite_model}")
        logger.info(f"  AI 浓度档位 : {self.cfg.ai_ratios}")
        logger.info(f"  混合模式    : {self.cfg.mixing_modes}")
        logger.info(f"  断点恢复    : {self._checkpoint.completed_count()} 条已完成")
        logger.info(f"  输出路径    : {self._output_path}")
        logger.info("=" * 60)

        # Step 1: 加载预采样人类文本
        documents = load_human_texts(
            self.cfg.source_path,
            shuffle=True,
            max_count=max_docs,
            seed=self.cfg.random_seed,
        )
        if not documents:
            logger.error("没有符合条件的文档，请检查数据路径和过滤参数。")
            return

        # Step 2: 生成所有变体任务
        all_tasks = self._build_tasks(documents)
        pending = [t for t in all_tasks if not self._checkpoint.is_completed(t.record_id)]
        pending = self._reconcile_with_existing_output(pending)
        logger.info(
            f"任务概况: 总变体 {len(all_tasks)}, "
            f"已完成 {len(all_tasks) - len(pending)}, "
            f"待处理 {len(pending)}"
        )

        if not pending:
            logger.info("所有任务已完成，无需重新生成。")
            self._print_summary(len(all_tasks))
            return

        # Step 3-5: 并发处理
        rewriter = None if dry_run else create_rewriter(self.cfg)

        with JsonlWriter(self._output_path) as writer:
            await self._process_all(pending, rewriter, writer, dry_run=dry_run)

        self._print_summary(len(all_tasks))

    # ------------------------------------------------------------------
    # 断点修复
    # ------------------------------------------------------------------

    def _reconcile_with_existing_output(self, pending: list[VariantTask]) -> list[VariantTask]:
        """
        启动时对账 output JSONL 与 checkpoint。
        若 output 已存在某 record_id 但 checkpoint 缺失，则补记 checkpoint 并从 pending 移除。
        """
        existing_ids = load_existing_record_ids(self._output_path)
        if not existing_ids:
            return pending
        existing_variant_keys = {record_id_to_variant_key(rid) for rid in existing_ids}

        reconciled = 0
        still_pending: list[VariantTask] = []
        for task in pending:
            task_variant_key = record_id_to_variant_key(task.record_id)
            if task.record_id in existing_ids or task_variant_key in existing_variant_keys:
                self._checkpoint.mark_completed(task.record_id, source_doc_id=task.doc.doc_id)
                reconciled += 1
            else:
                still_pending.append(task)

        if reconciled:
            logger.info(
                f"断点修复: 从输出文件对账补记 {reconciled} 条，避免重复生成。"
            )
        return still_pending

    # ------------------------------------------------------------------
    # 任务生成
    # ------------------------------------------------------------------

    def _build_tasks(self, documents: list[SourceDocument]) -> list[VariantTask]:
        """
        为每篇文档生成所有 ratio 变体的 VariantTask。

        每篇文档只分配一种混合模式（block_replace 或 random_scatter），
        两种模式在文档级别保持约 1:1（奇数篇时数量差最多 1）。

        重要：模式分配不依赖 max_docs 总量，保证断点续跑时同一文档的
        record_id（含 mode 后缀）稳定可复用。
        """
        tasks: list[VariantTask] = []
        modes = list(dict.fromkeys(self.cfg.mixing_modes))
        if not modes:
            modes = ["block_replace", "random_scatter"]

        # 与 max_docs 无关的稳定分配：按顺序轮转模式。
        # documents 已在 load_human_texts 中按固定 seed 打乱，因此该策略可复现且前缀稳定。
        mode_offset = self.cfg.random_seed % len(modes)
        for doc_idx, doc in enumerate(documents):
            assigned_mode = modes[(doc_idx + mode_offset) % len(modes)]
            # 使用稳定哈希生成文档级随机种子，保证跨进程可复现。
            doc_seed = int(
                hashlib.md5(f"{self.cfg.random_seed}:{doc.doc_id}".encode("utf-8")).hexdigest()[:8],
                16,
            )
            doc_rng = random.Random(doc_seed)
            variants = enumerate_variants(
                doc.text,
                self.cfg.ai_ratios,
                [assigned_mode],
                doc_rng,
            )
            for sel in variants:
                rid = make_record_id(doc.doc_id, sel.target_ratio, sel.mode)
                tasks.append(VariantTask(doc=doc, selection=sel, record_id=rid))
        return tasks

    # ------------------------------------------------------------------
    # 并发处理主循环
    # ------------------------------------------------------------------

    async def _process_all(
        self,
        tasks: list[VariantTask],
        rewriter: BaseLLMRewriter | None,
        writer: JsonlWriter,
        dry_run: bool,
    ) -> None:
        """
        以 asyncio.gather + Semaphore 并发执行所有 VariantTask。
        实时打印进度条（含 ETA、token 与费用估算）。
        """
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

            ratio = (processed_count / total) if total else 1.0
            width = 28
            done = int(width * ratio)
            bar = f"[{'#' * done}{'-' * (width - done)}]"

            elapsed = max(0.001, now - start_ts)
            speed = processed_count / elapsed
            remaining = max(0, total - processed_count)
            eta = int(remaining / speed) if speed > 0 else -1
            eta_str = f"{eta}s" if eta >= 0 else "N/A"

            cost_str = f"${self._estimated_cost_usd:.4f}"

            logger.info(
                f"进度 {bar} {ratio * 100:6.2f}% "
                f"{processed_count}/{total} | 成功 {success_count} | 跳过 {skipped_count} | "
                f"{speed:.2f} task/s | ETA {eta_str} | "
                f"tokens in={self._total_input_tokens}, out={self._total_output_tokens} | 费用 {cost_str}"
            )
            last_render_ts = now

        async def _process_one(task: VariantTask) -> None:
            nonlocal processed_count, success_count, skipped_count
            async with semaphore:
                result = await self._process_task(task, rewriter, dry_run)
            if result is not None:
                record, in_tok, out_tok = result
                writer.write(record)
                req_cost = estimate_request_cost_usd(self.cfg.rewrite_model, in_tok, out_tok)
                if req_cost is not None:
                    self._estimated_cost_usd += req_cost
                self._checkpoint.mark_completed(
                    task.record_id,
                    source_doc_id=task.doc.doc_id,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    estimated_cost_usd=req_cost or 0.0,
                )
                self._total_input_tokens += in_tok
                self._total_output_tokens += out_tok
                success_count += 1
            else:
                skipped_count += 1
            processed_count += 1
            _render_progress(force=(processed_count == total))

        await asyncio.gather(*[_process_one(t) for t in tasks])

    # ------------------------------------------------------------------
    # 单任务处理
    # ------------------------------------------------------------------

    async def _process_task(
        self,
        task: VariantTask,
        rewriter: BaseLLMRewriter | None,
        dry_run: bool,
    ) -> tuple[DatasetRecord, int, int] | None:
        """处理单个 VariantTask，返回 (DatasetRecord, in_tok, out_tok) 或 None（跳过）。"""
        sel = task.selection
        doc = task.doc

        try:
            # ---- ratio = 0.0：纯人类基线，无需调用 API ----
            if sel.target_ratio == 0.0 or not sel.selected_indices:
                rewrites: dict[int, str] = {}
                in_tok, out_tok = 0, 0

            # ---- dry_run：跳过 API，直接用原句 ----
            elif dry_run:
                rewrites = {i: sel.sentences[i] for i in sel.selected_indices}
                in_tok, out_tok = 0, 0

            # ---- 正常调用 LLM ----
            else:
                result = await rewriter.rewrite(  # type: ignore[union-attr]
                    sel.sentences, sel.selected_indices,
                    task_id=task.record_id,
                )
                rewrites = result.rewrites
                in_tok, out_tok = result.input_tokens, result.output_tokens

            # ---- 改写失败/不完整：跳过该样本，避免污染标签 ----
            if (not dry_run) and sel.target_ratio > 0.0 and sel.selected_indices:
                expected = len(sel.selected_indices)
                got = len(rewrites)
                if got < expected:
                    logger.warning(
                        f"[{task.record_id}] 改写不完整（expected={expected}, got={got}），"
                        "该样本已跳过，不写入数据集。"
                    )
                    return None

            # ---- 回填 ----
            mixed_sentences = sel.build_mixed_sentences(rewrites)
            mixed_text = " ".join(mixed_sentences)
            sentence_labels = sel.sentence_label_array(rewrites)

            # ---- 精确标签计算 ----
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
                mixed_text=mixed_text,
                n_sentences=sel.n,
                target_ai_ratio=sel.target_ratio,
                mixing_mode=sel.mode,
                rewrite_model=self.cfg.rewrite_model if (not dry_run and sel.target_ratio > 0.0) else ("human" if sel.target_ratio == 0.0 else "dry_run"),
                sentence_labels=sentence_labels,
                lir=label_dict.get("lir", 0.0),
                jaccard_distance=label_dict.get("jaccard_distance"),
                cosine_distance=label_dict.get("cosine_distance"),
            ), in_tok, out_tok

        except (NonRetryableAPIError, RetryExhaustedAPIError) as exc:
            logger.error(
                f"[{task.record_id}] API 错误不可恢复，程序终止: "
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
        est_cost = estimate_cost_usd(
            self.cfg.rewrite_model,
            stats.get("api_input_tokens", 0),
            stats.get("api_output_tokens", 0),
        )
        logger.info("=" * 60)
        logger.info("Pipeline 完成")
        logger.info(f"  总变体数        : {total_tasks}")
        logger.info(f"  成功写入        : {stats.get('total_written', 0)}")
        logger.info(f"  原始文本数      : {stats.get('total_source_docs_processed', 0)}")
        logger.info(f"  API 输入 Tokens : {stats.get('api_input_tokens', 0):,}")
        logger.info(f"  API 输出 Tokens : {stats.get('api_output_tokens', 0):,}")
        cost_from_checkpoint = float(stats.get("api_estimated_cost_usd", est_cost))
        if cost_from_checkpoint >= 0:
            breakdown = estimate_cost_breakdown_usd(
                self.cfg.rewrite_model,
                stats.get("api_input_tokens", 0),
                stats.get("api_output_tokens", 0),
            )
            if breakdown is not None:
                if self.cfg.rewrite_model in {"qwen3.5-plus", "qwen3.5-flash", "qwen3.5-flash-2026-02-23"}:
                    logger.info("  API 费用预估公式: 对每次请求按 input_tokens 分档取单价，再累加")
                else:
                    logger.info("  API 费用预估公式: input_tokens/1e6*输入单价 + output_tokens/1e6*输出单价")
                logger.info(
                    f"  单价(USD/1M)    : in={breakdown['input_rate_per_1m']}, "
                    f"out={breakdown['output_rate_per_1m']}"
                )
                logger.info(
                    f"  费用拆分(USD)   : in={breakdown['input_cost_usd']:.4f}, "
                    f"out={breakdown['output_cost_usd']:.4f}"
                )
            logger.info(f"  API 费用预估    : ${cost_from_checkpoint:.4f}")
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
            f"  Cosine Distance:  {self.labels.get('cosine_distance', 'N/A')}",
            f"  doc_ai_ratio:     {self.labels.get('doc_ai_ratio_exact', 'N/A')}",
            "",
            f"API Tokens: in={self.input_tokens}, out={self.output_tokens}",
        ]
        return "\n".join(lines)

    def to_dataset_record(self, record_id: str = "", source_dataset: str = "single_test", domain: str = "test") -> DatasetRecord:
        """转换为与批量 Pipeline 完全一致的 DatasetRecord 格式。"""
        if not record_id:
            import hashlib
            text_hash = hashlib.md5(self.original_text.encode()).hexdigest()[:8]
            record_id = make_record_id(f"single_{text_hash}", self.target_ratio, self.mixing_mode)

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
            cosine_distance=self.labels.get("cosine_distance"),
        )


async def process_single_text(
    text: str,
    target_ratio: float = 0.4,
    mixing_mode: str = "block_replace",
    model: str = "qwen3.5-plus",
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
    from .sentence_processor import split_into_sentences, create_sentence_selection

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
        result = await rewriter.rewrite(sentences, selection.selected_indices, language_hint)
        rewrites = result.rewrites
        in_tok, out_tok = result.input_tokens, result.output_tokens

    # Step 4: 回填
    mixed_sentences = selection.build_mixed_sentences(rewrites)
    mixed_text = " ".join(mixed_sentences)
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
    model: str = "qwen3.5-plus",
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
            model="qwen3.5-plus",
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
