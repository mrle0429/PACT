"""
数据集写入模块：JSONL 输出 + 断点续传（Checkpoint）支持。
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .utils import get_logger

logger = get_logger(__name__)


def record_id_to_variant_key(record_id: str) -> str:
    """
    将 record_id 归一化为“忽略 mode”的变体键。
    例如: arxiv_2408.17298_r80_block -> arxiv_2408.17298_r80
    """
    rid = str(record_id)
    if rid.endswith("_block"):
        return rid[:-6]
    if rid.endswith("_scatter"):
        return rid[:-8]
    return rid


# ---------------------------------------------------------------------------
# 输出记录数据类（扁平化结构）
# ---------------------------------------------------------------------------

@dataclass
class DatasetRecord:
    """
    一条完整的混合数据集记录（扁平化，所有字段一级展开）。

    样例::

        {
          "id":               "arxiv_1409.3719_r40_block",
          "source_dataset":   "arxiv",
          "source_domain":    "academic",
          "original_text":    "...",
          "mixed_text":       "...",
          "n_sentences":      6,
          "target_ai_ratio":  0.4,
          "mixing_mode":      "random_scatter",
          "rewrite_model":    "qwen3.5-plus",
          "sentence_labels":  [1, 0, 1, 0, 0, 0],
          "lir":              0.3822,
          "jaccard_distance": 0.2451,
          "cosine_distance":  0.1893,
          "extra":            {}
        }
    """

    # ---- 标识 ----
    id: str
    source_dataset: str
    source_domain: str

    # ---- 文本 ----
    original_text: str = ""
    mixed_text: str = ""

    # ---- 生成参数 ----
    n_sentences: int = 0
    target_ai_ratio: float = 0.0
    mixing_mode: str = ""
    rewrite_model: str = ""

    # ---- 标签 ----
    sentence_labels: list[int] = field(default_factory=list)
    lir: float = 0.0
    jaccard_distance: float | None = None
    cosine_distance: float | None = None

    # ---- 扩展（不同数据集来源的特有字段）----
    extra: dict = field(default_factory=dict)

    def to_jsonl_line(self) -> str:
        """序列化为单行 JSON 字符串（无换行）。"""
        return json.dumps(asdict(self), ensure_ascii=False)


# ---------------------------------------------------------------------------
# JSONL Writer（流式追加，支持大数据量）
# ---------------------------------------------------------------------------

class JsonlWriter:
    """
    流式 JSONL 写入器。
    - 每条记录逐行写入，避免 OOM
    - 自动创建父目录
    - 调用 close() 或作为上下文管理器使用
    """

    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._existing_ids = load_existing_record_ids(self.filepath)
        self._fh = self.filepath.open("a", encoding="utf-8")
        self._count = 0
        self._skipped_duplicates = 0

    def write(self, record: DatasetRecord) -> bool:
        # 幂等写入：若记录已存在则跳过，避免断点恢复时重复行。
        if record.id in self._existing_ids:
            self._skipped_duplicates += 1
            return False

        self._fh.write(record.to_jsonl_line() + "\n")
        self._fh.flush()
        self._count += 1
        self._existing_ids.add(record.id)
        return True

    def close(self) -> None:
        self._fh.close()
        logger.info(
            f"已写入 {self._count} 条记录，跳过重复 {self._skipped_duplicates} 条 → {self.filepath}"
        )

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# 断点续传：Checkpoint 管理
# ---------------------------------------------------------------------------

class CheckpointManager:
    """
    记录已完成的文档 ID，支持中断后从断点继续。
    状态文件为简单的 JSON（{completed_ids: [...], stats: {...}}）。
    """

    def __init__(self, checkpoint_dir: str | Path, run_name: str):
        self._dir = Path(checkpoint_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / f"{run_name}.json"
        self._state = self._load()

    def _load(self) -> dict:
        def _empty_state() -> dict:
            return {
                "completed_ids": [],
                "completed_source_ids": [],
                "stats": {
                    "total_written": 0,
                    "total_source_docs_processed": 0,
                    "api_input_tokens": 0,
                    "api_output_tokens": 0,
                    "api_estimated_cost_usd": 0.0,
                },
            }

        if self._path.exists():
            try:
                with self._path.open(encoding="utf-8") as fh:
                    state = json.load(fh)
                # 向后兼容旧 checkpoint：补齐新增统计字段
                stats = state.setdefault("stats", {})
                stats.setdefault("total_written", 0)
                stats.setdefault("api_input_tokens", 0)
                stats.setdefault("api_output_tokens", 0)
                stats.setdefault("api_estimated_cost_usd", 0.0)
                state.setdefault("completed_ids", [])
                state.setdefault("completed_source_ids", [])
                # 向后兼容：旧 checkpoint 缺失该统计时，按 completed_source_ids 重建
                stats.setdefault(
                    "total_source_docs_processed",
                    len(set(state.get("completed_source_ids", []))),
                )
                n = len(state.get("completed_ids", []))
                logger.info(f"加载断点: 已完成 {n} 条 ({self._path})")
                return state
            except Exception as exc:
                logger.warning(f"断点文件读取失败，将从头开始: {exc}")
        return _empty_state()

    def _save(self) -> None:
        tmp = self._path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(self._state, fh, indent=2)
        tmp.replace(self._path)

    def is_completed(self, record_id: str) -> bool:
        if record_id in self._completed_set:
            return True
        return record_id_to_variant_key(record_id) in self._completed_variant_key_set

    def mark_completed(
        self,
        record_id: str,
        source_doc_id: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        estimated_cost_usd: float = 0.0,
    ) -> None:
        if record_id not in self._completed_set:
            self._state["completed_ids"].append(record_id)
            self._completed_set.add(record_id)
            self._completed_variant_key_set.add(record_id_to_variant_key(record_id))
            if source_doc_id and source_doc_id not in self._completed_source_set:
                self._state["completed_source_ids"].append(source_doc_id)
                self._completed_source_set.add(source_doc_id)
                self._state["stats"]["total_source_docs_processed"] += 1
            self._state["stats"]["total_written"] += 1
            self._state["stats"]["api_input_tokens"] += input_tokens
            self._state["stats"]["api_output_tokens"] += output_tokens
            self._state["stats"]["api_estimated_cost_usd"] += estimated_cost_usd
            self._save()

    @property
    def _completed_set(self) -> set:
        # 缓存到 _set 属性避免每次重建
        if not hasattr(self, "_completed_id_set"):
            self._completed_id_set: set[str] = set(self._state["completed_ids"])
        return self._completed_id_set

    @property
    def _completed_variant_key_set(self) -> set:
        if not hasattr(self, "_completed_variant_set"):
            self._completed_variant_set: set[str] = {
                record_id_to_variant_key(rid) for rid in self._state["completed_ids"]
            }
        return self._completed_variant_set

    @property
    def _completed_source_set(self) -> set:
        if not hasattr(self, "_completed_source_id_set"):
            self._completed_source_id_set: set[str] = set(
                self._state.get("completed_source_ids", [])
            )
        return self._completed_source_id_set

    @property
    def stats(self) -> dict:
        return dict(self._state["stats"])

    def completed_count(self) -> int:
        return len(self._state["completed_ids"])


# ---------------------------------------------------------------------------
# 辅助：生成记录 ID
# ---------------------------------------------------------------------------

def make_record_id(
    doc_id: str,
    target_ratio: float,
    mixing_mode: str,
) -> str:
    """
    生成唯一记录 ID。

    格式: "{doc_id}_r{ratio}_{mode}"
    示例: "arxiv_1409.3719_r40_block"
    """
    ratio_tag = f"r{int(target_ratio * 100)}"
    mode_tag = "block" if "block" in mixing_mode else "scatter"
    safe_id = str(doc_id).replace("/", "-")[:40]
    return f"{safe_id}_{ratio_tag}_{mode_tag}"


def load_existing_record_ids(filepath: str | Path) -> set[str]:
    """
    从既有 JSONL 中加载已存在的 record_id 集合。
    用于断点恢复时幂等写入与 checkpoint 修复。
    """
    path = Path(filepath)
    if not path.exists():
        return set()

    ids: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                logger.warning(f"输出文件第 {lineno} 行 JSON 解析失败，已跳过。")
                continue
            rid = obj.get("id")
            if isinstance(rid, str) and rid:
                ids.add(rid)
    return ids
