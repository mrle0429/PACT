"""
数据集写入模块：JSONL 输出 + 运行状态（checkpoint）支持。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .utils import get_logger

logger = get_logger(__name__)

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
          "sentence_jaccard": 0.2113,
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
    sentence_jaccard: float | None = None
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

    def __init__(self, filepath: str | Path, existing_ids: set[str] | None = None):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._existing_ids = (
            set(existing_ids)
            if existing_ids is not None
            else load_existing_record_ids(self.filepath)
        )
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
# 运行状态：Checkpoint 管理
# ---------------------------------------------------------------------------

class CheckpointManager:
    """
    记录与当前输出文件对应的运行状态。

    断点续传的唯一来源是最终输出 JSONL；checkpoint 仅保存：
    - 与输出文件同步的统计快照
    - 无法从输出文件反推的运行统计（如 API token 消耗）
    """

    def __init__(self, checkpoint_dir: str | Path, run_name: str):
        self._dir = Path(checkpoint_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / f"{run_name}.json"
        self._state = self._load()

    def _load(self) -> dict:
        def _empty_state() -> dict:
            return {
                "output_source_ids": [],
                "stats": {
                    "total_written": 0,
                    "total_source_docs_processed": 0,
                    "api_input_tokens": 0,
                    "api_output_tokens": 0,
                },
            }

        if self._path.exists():
            try:
                with self._path.open(encoding="utf-8") as fh:
                    state = json.load(fh)
                stats = state.setdefault("stats", {})
                stats.setdefault("total_written", 0)
                stats.setdefault("api_input_tokens", 0)
                stats.setdefault("api_output_tokens", 0)
                output_source_ids = state.get("output_source_ids")
                if output_source_ids is None:
                    output_source_ids = state.get("completed_source_ids", [])
                state["output_source_ids"] = output_source_ids
                stats.setdefault(
                    "total_source_docs_processed",
                    len(set(output_source_ids)),
                )
                logger.info(
                    "加载运行状态: 已写入 %s 条 (%s)",
                    stats["total_written"],
                    self._path,
                )
                return state
            except Exception as exc:
                logger.warning(f"运行状态文件读取失败，将从头开始: {exc}")
        return _empty_state()

    def _save(self) -> None:
        tmp = self._path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(self._state, fh, indent=2)
        tmp.replace(self._path)

    def sync_output_snapshot(
        self,
        total_written: int,
        source_doc_ids: list[str],
    ) -> bool:
        """
        用当前输出文件快照刷新运行状态。

        `total_written` 和 `source_doc_ids` 都由输出文件推导。
        """
        stats = self._state.setdefault("stats", {})
        normalized_source_ids = list(dict.fromkeys(source_doc_ids))
        current_written = int(stats.get("total_written", 0))
        if (
            current_written == total_written
            and self._state.get("output_source_ids", []) == normalized_source_ids
        ):
            return False

        if total_written < current_written:
            logger.warning("检测到输出文件记录数回退，已重置 API token 统计。")
            stats["api_input_tokens"] = 0
            stats["api_output_tokens"] = 0

        stats["total_written"] = total_written
        stats["total_source_docs_processed"] = len(normalized_source_ids)
        self._state["output_source_ids"] = normalized_source_ids
        self._output_source_id_set = set(normalized_source_ids)
        self._save()
        return True

    def record_write(
        self,
        source_doc_id: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        stats = self._state.setdefault("stats", {})
        stats["total_written"] += 1
        if source_doc_id and source_doc_id not in self._output_source_set:
            self._state["output_source_ids"].append(source_doc_id)
            self._output_source_set.add(source_doc_id)
            stats["total_source_docs_processed"] += 1
        stats["api_input_tokens"] += input_tokens
        stats["api_output_tokens"] += output_tokens
        self._save()

    @property
    def _output_source_set(self) -> set[str]:
        if not hasattr(self, "_output_source_id_set"):
            self._output_source_id_set = set(self._state.get("output_source_ids", []))
        return self._output_source_id_set

    @property
    def stats(self) -> dict:
        return dict(self._state["stats"])


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
    用于断点恢复时的幂等写入与 pending 计算。
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
