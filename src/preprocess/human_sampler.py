"""
人类文本采样模块 — 从四类数据集中采样原始人类文本。

支持的数据集：
  - ArXiv        (nick007x/arxiv-papers)          → abstract 字段
  - OpenWebText  (Skylion007/openwebtext)          → text 字段（streaming）
  - XSum         (EdinburghNLP/xsum)               → document 字段
  - DAIGT-v2     (thedrcat/daigt-v2-train-dataset) → text 字段（kagglehub, label=0 人类文本）

输出格式（JSONL）：
  {"id": "arxiv_0902.3253", "text": "...", "sentence_count": 12}
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import pysbd

from ..utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 采样源配置
# ---------------------------------------------------------------------------

@dataclass
class SourceConfig:
    """单个数据源的采样配置。"""
    hf_dataset_id: str          # HuggingFace 数据集标识（或 kaggle 数据集标识）
    text_field: str             # 文本字段名
    id_field: str | None        # 原始 ID 字段名（None 则自动生成）
    id_prefix: str              # ID 前缀，如 "arxiv"
    quota: int                  # 采样配额
    streaming: bool = False     # 是否使用流式加载
    split: str = "train"        # 数据集 split
    loader: str = "hf"          # 加载方式: "hf" (HuggingFace) | "kaggle_csv" (kagglehub)
    label_field: str | None = None   # 标签过滤字段（仅保留 label_value 的行）
    label_value: int | None = None   # 保留的标签值


# 四个预定义数据源（quota 后续由主函数设置）
DEFAULT_SOURCES: dict[str, SourceConfig] = {
    "arxiv": SourceConfig(
        hf_dataset_id="nick007x/arxiv-papers",
        text_field="abstract",
        id_field="arxiv_id",
        id_prefix="arxiv",
        quota=2500,
    ),
    "owt": SourceConfig(
        hf_dataset_id="Skylion007/openwebtext",
        text_field="text",
        id_field=None,          # 无原生 ID，使用行号
        id_prefix="owt",
        quota=2500,
        streaming=True,         # 数据集过大，必须流式
    ),
    "xsum": SourceConfig(
        hf_dataset_id="EdinburghNLP/xsum",
        text_field="document",
        id_field="id",
        id_prefix="xsum",
        quota=2500,
    ),
    "daigt": SourceConfig(
        hf_dataset_id="thedrcat/daigt-v2-train-dataset",
        text_field="text",
        id_field=None,          # 无原生唯一 ID，使用 DataFrame 行索引
        id_prefix="daigt",
        quota=2500,
        loader="kaggle_csv",    # 通过 kagglehub 下载 CSV
        label_field="label",    # 仅保留人类文本
        label_value=0,
    ),
}


# ---------------------------------------------------------------------------
# 过滤配置
# ---------------------------------------------------------------------------

@dataclass
class FilterConfig:
    """通用文本过滤参数。"""
    min_sentences: int = 8
    max_sentences: int = 20
    min_words: int = 100
    max_words: int = 300
    max_chars: int = 8000
    single_sentence_min_words: int = 5
    single_sentence_max_words: int = 50
    random_seed: int = 2006


# ---------------------------------------------------------------------------
# 文本清洗
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# EnronEmail 清洗正则（预编译）
# ---------------------------------------------------------------------------

def clean_arxiv_abstract(text: str) -> str:
    """清洗 ArXiv 摘要：去除 LaTeX 残留。"""
    # 去除 \cite{...}, \ref{...} 等命令
    text = re.sub(r"\\(?:cite|ref|eqref|label)\{[^}]*\}", "", text)
    # 简单处理行内公式标记（保留内容）
    text = re.sub(r"\$([^$]+)\$", r"\1", text)
    # 去除残余反斜杠命令（如 \emph{word} → word）
    text = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", text)
    # 去除多余空格
    text = re.sub(r"  +", " ", text)
    return text.strip()


def _normalize_whitespace(text: str) -> str:
    """将换行符替换为空格，并合并连续空白。"""
    text = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    text = re.sub(r"  +", " ", text)
    return text.strip()


def clean_text(text: str, source: str) -> str:
    """根据数据源名称分派清洗逻辑（统一执行换行符清洗）。"""
    if source == "arxiv":
        text = clean_arxiv_abstract(text)
    else:
        text = text.strip()
    # 统一换行符清洗
    text = _normalize_whitespace(text)
    return text


# ---------------------------------------------------------------------------
# 分句器（PySBD）
# ---------------------------------------------------------------------------

_segmenter = pysbd.Segmenter(language="en", clean=False)


def _pysbd_sentences(text: str) -> list[str]:
    """使用 PySBD 分句，返回非空句子列表。"""
    return [s.strip() for s in _segmenter.segment(text) if s.strip()]


# ---------------------------------------------------------------------------
# 过滤逻辑
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")


def _count_words(text: str) -> int:
    """统计文本词数。"""
    return len(_WORD_RE.findall(text))


def passes_filter(text: str, cfg: FilterConfig) -> bool:
    """检查文本是否通过字数/句数/单句长度过滤。"""
    if not text:
        return False
    if len(text) > cfg.max_chars:
        return False

    total_words = _count_words(text)
    if not (cfg.min_words <= total_words <= cfg.max_words):
        return False

    try:
        sentences = _pysbd_sentences(text)
    except Exception:
        return False

    n_sentences = len(sentences)
    if not (cfg.min_sentences <= n_sentences <= cfg.max_sentences):
        return False

    for sent in sentences:
        sent_words = _count_words(sent)
        if not (cfg.single_sentence_min_words <= sent_words <= cfg.single_sentence_max_words):
            return False
    return True


# ---------------------------------------------------------------------------
# ID 生成
# ---------------------------------------------------------------------------

def make_id(prefix: str, raw_id: str | None, index: int) -> str:
    """
    构造唯一 ID。

    - 有原生 ID 且无特殊字符：{prefix}_{raw_id}
    - 有原生 ID 但含特殊字符（如 EnronEmail 的 message_id）：
      {prefix}_{md5_hex[:16]}
    - 无原生 ID：{prefix}_{index:08d}
    """
    if raw_id is None:
        return f"{prefix}_{index:08d}"

    raw_id = str(raw_id).strip()
    # 检查是否含有文件名/ID 不友好的字符
    if re.search(r"[<>@\s/\\]", raw_id):
        digest = hashlib.md5(raw_id.encode()).hexdigest()[:16]
        return f"{prefix}_{digest}"
    return f"{prefix}_{raw_id}"


# ---------------------------------------------------------------------------
# 单数据源采样
# ---------------------------------------------------------------------------

def _load_kaggle_csv(src: SourceConfig, seed: int) -> Iterator:
    """
    通过 kagglehub 下载 Kaggle 数据集（CSV），
    按标签过滤并 shuffle 后逐行产出 dict。
    """
    import kagglehub
    import pandas as pd
    import random as _random

    logger.info(f"[{src.id_prefix}] 通过 kagglehub 下载 {src.hf_dataset_id} ...")
    dataset_path = Path(kagglehub.dataset_download(src.hf_dataset_id))

    # 找到 CSV 文件
    csv_files = list(dataset_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"kagglehub 下载目录中无 CSV 文件: {dataset_path}")
    csv_path = csv_files[0]
    logger.info(f"[{src.id_prefix}] 读取 {csv_path.name}")

    df = pd.read_csv(csv_path)
    logger.info(f"[{src.id_prefix}] 原始行数: {len(df)}")

    # 按标签过滤（仅保留人类文本）
    if src.label_field is not None and src.label_value is not None:
        df = df[df[src.label_field] == src.label_value].copy()
        logger.info(
            f"[{src.id_prefix}] 标签过滤 ({src.label_field}=={src.label_value}): "
            f"剩余 {len(df)} 条"
        )

    # shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    for idx, row in df.iterrows():
        yield dict(row) | {"_row_idx": idx}


def sample_from_source(
    src: SourceConfig,
    filter_cfg: FilterConfig,
) -> list[dict]:
    """
    从单个数据源采样指定数量的人类文本。
    支持 HuggingFace datasets 和 kagglehub CSV 两种加载方式。

    返回: [{"id": str, "text": str}, ...]
    """
    # 根据 loader 类型选择数据迭代器
    if src.loader == "kaggle_csv":
        data_iter = _load_kaggle_csv(src, filter_cfg.random_seed)
    else:
        from datasets import load_dataset

        logger.info(
            f"[{src.id_prefix}] 开始加载 {src.hf_dataset_id} "
            f"(split={src.split}, streaming={src.streaming})"
        )

        ds = load_dataset(
            src.hf_dataset_id,
            split=src.split,
            streaming=src.streaming,
        )

        # 非流式模式：先 shuffle 整个数据集
        if not src.streaming:
            ds = ds.shuffle(seed=filter_cfg.random_seed)
        else:
            # 流式模式：使用 buffer shuffle
            ds = ds.shuffle(seed=filter_cfg.random_seed, buffer_size=20_000)

        data_iter = iter(ds)

    results: list[dict] = []
    seen_ids: set[str] = set()
    scanned = 0
    skipped_empty = 0
    skipped_filter = 0

    for idx, row in enumerate(data_iter):
        if len(results) >= src.quota:
            break

        scanned += 1

        # 提取文本
        text = str(row.get(src.text_field, "") or "").strip()
        if not text:
            skipped_empty += 1
            continue

        # 清洗
        text = clean_text(text, src.id_prefix)
        if not text:
            skipped_empty += 1
            continue

        # 过滤
        if not passes_filter(text, filter_cfg):
            skipped_filter += 1
            continue
        sentences = _pysbd_sentences(text)
        n_sentences = len(sentences)

        # 构造 ID（不放回：检查去重）
        raw_id = row.get(src.id_field) if src.id_field else None
        row_idx = row.get("_row_idx", idx)  # kaggle_csv 模式使用原始行索引
        doc_id = make_id(src.id_prefix, raw_id, row_idx)
        if doc_id in seen_ids:
            continue
        seen_ids.add(doc_id)

        results.append({"id": doc_id, "text": text, "sentence_count": n_sentences})

        # 进度日志（每 500 条）
        if len(results) % 500 == 0:
            logger.info(
                f"[{src.id_prefix}] 已采样 {len(results)}/{src.quota} 条 "
                f"(扫描 {scanned} 条)"
            )

    logger.info(
        f"[{src.id_prefix}] 采样完成: {len(results)}/{src.quota} 条 | "
        f"扫描 {scanned} | 空文本 {skipped_empty} | 过滤掉 {skipped_filter}"
    )

    if len(results) < src.quota:
        logger.warning(
            f"[{src.id_prefix}] 警告：仅采样到 {len(results)} 条，"
            f"不足目标 {src.quota} 条！"
        )

    return results


# ---------------------------------------------------------------------------
# 主采样流程
# ---------------------------------------------------------------------------

def build_human_dataset(
    output_path: str = "output/human_texts_10k.jsonl",
    total: int = 10_000,
    seed: int = 42,
    sources: dict[str, SourceConfig] | None = None,
    filter_cfg: FilterConfig | None = None,
) -> Path:
    """
    从各数据源等比采样人类文本，合并写出为 JSONL。

    Args:
        output_path: 输出文件路径
        total: 总采样数（将平均分配到各数据源）
        seed: 随机种子
        sources: 数据源配置（默认使用 DEFAULT_SOURCES）
        filter_cfg: 过滤配置

    Returns:
        输出文件的 Path
    """
    import random

    if sources is None:
        sources = {k: SourceConfig(**v.__dict__) for k, v in DEFAULT_SOURCES.items()}

    if filter_cfg is None:
        filter_cfg = FilterConfig(random_seed=seed)

    # 平分配额
    n_sources = len(sources)
    quota_per_source = total // n_sources
    remainder = total % n_sources

    for i, (name, src) in enumerate(sources.items()):
        src.quota = quota_per_source + (1 if i < remainder else 0)

    logger.info(
        f"采样计划: 总计 {total} 条, {n_sources} 个数据源, "
        f"每源 {quota_per_source} 条, seed={seed}"
    )

    # 依次采样
    all_samples: list[dict] = []
    for name, src in sources.items():
        samples = sample_from_source(src, filter_cfg)
        all_samples.extend(samples)

    # 全局 shuffle
    rng = random.Random(seed)
    rng.shuffle(all_samples)

    # 写出
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for record in all_samples:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"数据集已写出: {out} ({len(all_samples)} 条)")

    # 统计摘要
    from collections import Counter
    prefix_counts = Counter(r["id"].split("_")[0] for r in all_samples)
    for prefix, count in sorted(prefix_counts.items()):
        logger.info(f"  {prefix}: {count} 条")

    return out
