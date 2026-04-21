"""
数据加载模块 — 从预采样的 JSONL 文件加载人类文本。

预采样JSONL文件包含以下字段：
    - id: 文档唯一标识符（通常包含来源前缀）
    - text: 文本内容
    - sentence_count: 句子数量
    - fixed_mixing_mode: 固定混合模式(Bolock Replace / Random Scatter)

"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from .sentence_processor import MixingMode
from .utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 领域数据类
# ---------------------------------------------------------------------------

@dataclass
class SourceDocument:
    """标准化的人类种子文档。"""
    doc_id: str
    text: str
    domain: str
    source_dataset: str
    sentence_count: int = 0
    fixed_mixing_mode: MixingMode | None = None


# ---------------------------------------------------------------------------
# ID 前缀 → (领域, 来源数据集) 映射
# ---------------------------------------------------------------------------

_PREFIX_META: dict[str, tuple[str, str]] = {
    "arxiv": ("academic", "arxiv"),
    "owt":   ("web",      "openwebtext"),
    "xsum":  ("news",     "xsum"),
    "daigt": ("essay",    "daigt"),
}


# ---------------------------------------------------------------------------
# 加载函数
# ---------------------------------------------------------------------------

def load_human_texts(
    path: str | Path = "data/human_texts_1k.cleaned.jsonl",
    *,
    shuffle: bool = True,
    max_count: int | None = None,
    seed: int = 42,
) -> list[SourceDocument]:
    """
    从预采样 JSONL 文件加载人类文本。

    数据已在采样阶段 (scripts/sample_human_texts.py) 完成以下处理：
      - 文本清洗（换行符归一化、LaTeX 残留清理）
      - 质量过滤（句子数 / 字符数范围）
      - 无放回抽样 + 去重

    Args:
        path:       JSONL 文件路径
        shuffle:    是否随机打乱
        max_count:  最多加载条数（None = 全部）
        seed:       随机种子

    Returns:
        SourceDocument 列表
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"人类文本文件不存在: {path.resolve()}")

    documents: list[SourceDocument] = []

    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning(f"第 {lineno} 行 JSON 解析失败，已跳过: {exc}")
                continue

            doc_id = str(record.get("id", f"line_{lineno}"))
            text = record.get("text", "")
            sentence_count = record.get("sentence_count", 0)
            fixed_mixing_mode = record.get("fixed_mixing_mode")

            if not text:
                continue

            # 从 ID 前缀推断领域和来源
            prefix = doc_id.split("_")[0] if "_" in doc_id else ""
            domain, source_ds = _PREFIX_META.get(prefix, ("unknown", "unknown"))

            documents.append(SourceDocument(
                doc_id=doc_id,
                text=text,
                domain=domain,
                source_dataset=source_ds,
                sentence_count=sentence_count,
                fixed_mixing_mode=fixed_mixing_mode,
            ))

    if shuffle:
        random.Random(seed).shuffle(documents)

    if max_count is not None:
        documents = documents[:max_count]

    logger.info(f"人类文本加载完成: {len(documents)} 条 (来源: {path.name})")
    return documents
