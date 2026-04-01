"""
analyze_checkpoint_mode.py

检查 checkpoint 中已完成任务的模式分配是否正确。

逻辑：
  - 用与 pipeline 完全相同的方式（固定 seed shuffle + 轮转）重算每个 doc_id 的期望模式
  - 从 record_id 后缀提取实际模式
  - 对比并报告不一致
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# 参数（与 pipeline 默认配置保持一致）
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = PROJECT_ROOT / "output" / "checkpoints" / "run_DeepSeek-V3.2_human_texts_1k.cleaned_3bfe49e14a.json"
SOURCE_JSONL   = PROJECT_ROOT / "data" / "human_texts_1k.jsonl"
RANDOM_SEED    = 42
MIXING_MODES   = ["block_replace", "random_scatter"]   # 顺序与 config 默认值相同


# ---------------------------------------------------------------------------
# Step 1：加载 checkpoint 中的已完成 record_id
# ---------------------------------------------------------------------------
def load_checkpoint(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data.get("completed_ids", [])


# ---------------------------------------------------------------------------
# Step 2：从 record_id 解析 doc_id 与实际 mode
# ---------------------------------------------------------------------------
def parse_record_id(rid: str) -> tuple[str, str | None]:
    """
    返回 (doc_id_without_ratio, actual_mode_tag)
    record_id 格式: {doc_id}_{ratio_tag}_{mode_tag}
    例: "owt_00002673_r40_scatter" -> doc_id_base="owt_00002673", mode="scatter"
    """
    if rid.endswith("_block"):
        mode_tag = "block"
        without_mode = rid[:-6]   # 去掉 "_block"
    elif rid.endswith("_scatter"):
        mode_tag = "scatter"
        without_mode = rid[:-8]   # 去掉 "_scatter"
    else:
        mode_tag = None
        without_mode = rid

    # 去掉 ratio tag（如 "_r40"）
    parts = without_mode.rsplit("_", 1)    # 从右分割一次
    if len(parts) == 2 and parts[1].startswith("r") and parts[1][1:].isdigit():
        doc_id = parts[0]
    else:
        doc_id = without_mode   # 无法识别，保留原样

    return doc_id, mode_tag


# ---------------------------------------------------------------------------
# Step 3：重建 pipeline 的期望模式分配
# ---------------------------------------------------------------------------
def build_expected_modes(
    source_jsonl: Path,
    random_seed: int,
    mixing_modes: list[str],
) -> dict[str, str]:
    """
    按 pipeline._build_tasks 完全相同的逻辑，返回 {doc_id: expected_mode_tag}。
    mode_tag: "block" | "scatter"
    """
    # 加载所有 doc_id（按文件顺序）
    doc_ids: list[str] = []
    with source_jsonl.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            doc_id = str(obj.get("id", f"line_{lineno}"))
            doc_ids.append(doc_id)

    # 用固定 seed 打乱 —— 与 load_human_texts(shuffle=True, seed=...) 完全一致
    shuffled = doc_ids.copy()
    random.Random(random_seed).shuffle(shuffled)

    modes = list(dict.fromkeys(mixing_modes))   # 去重保序
    mode_offset = random_seed % len(modes)

    expected: dict[str, str] = {}
    for doc_idx, doc_id in enumerate(shuffled):
        assigned_mode = modes[(doc_idx + mode_offset) % len(modes)]
        mode_tag = "block" if "block" in assigned_mode else "scatter"
        # 若同一 doc_id 出现多次（不应该），以第一次为准
        expected.setdefault(doc_id, mode_tag)

    return expected


# ---------------------------------------------------------------------------
# Step 4：对账与报告
# ---------------------------------------------------------------------------
def analyze(
    completed_ids: list[str],
    expected_modes: dict[str, str],
) -> None:
    # 收集每个 doc_id 在 checkpoint 中出现过的所有 mode_tag
    doc_actual_modes: dict[str, set[str]] = defaultdict(set)
    unknown_doc_ids: list[str] = []

    for rid in completed_ids:
        doc_id, mode_tag = parse_record_id(rid)
        if mode_tag is None:
            unknown_doc_ids.append(rid)
            continue
        doc_actual_modes[doc_id].add(mode_tag)

    # ------------------------------------------------------------------
    # 问题一：同一文档在 checkpoint 中出现了两种模式（历史残留错误）
    # ------------------------------------------------------------------
    dual_mode_docs: list[tuple[str, set[str]]] = [
        (doc_id, modes)
        for doc_id, modes in doc_actual_modes.items()
        if len(modes) > 1
    ]

    # ------------------------------------------------------------------
    # 问题二：只有单一模式 且 该模式与期望不符（排除双模式文档）
    # ------------------------------------------------------------------
    wrong_mode_docs: list[tuple[str, str, str]] = []
    not_in_expected: list[str] = []

    for doc_id, actual_modes in doc_actual_modes.items():
        if doc_id not in expected_modes:
            not_in_expected.append(doc_id)
            continue
        # 只考虑单一模式的文档：双模式文档中必然有一种符合期望，不计入问题二
        if len(actual_modes) > 1:
            continue
        expected_tag = expected_modes[doc_id]
        actual_tag = next(iter(actual_modes))
        if actual_tag != expected_tag:
            wrong_mode_docs.append((doc_id, actual_tag, expected_tag))

    # ------------------------------------------------------------------
    # 输出报告
    # ------------------------------------------------------------------
    total_docs   = len(doc_actual_modes)
    total_records = len(completed_ids)

    print("=" * 60)
    print("Checkpoint 模式一致性检查报告")
    print("=" * 60)
    print(f"checkpoint 中已完成记录数 : {total_records}")
    print(f"涉及唯一文档数            : {total_docs}")
    print(f"source JSONL 已知文档数   : {len(expected_modes)}")
    print()

    # ---- 双模式文档 ----
    print(f"[问题1] 同一文档被分配到两种模式 : {len(dual_mode_docs)} 篇")
    if dual_mode_docs:
        for doc_id, modes in sorted(dual_mode_docs):
            # 收集该文档在 checkpoint 中的所有 record_id
            related = [r for r in completed_ids if parse_record_id(r)[0] == doc_id]
            print(f"  - {doc_id}  实际模式: {sorted(modes)}")
            for r in sorted(related):
                print(f"      {r}")
    print()

    # ---- 模式与当前期望不符（仅单一模式文档）----
    print(f"[问题2] 单一模式但与期望不符（排除双模式文档）: {len(wrong_mode_docs)} 篇")
    if wrong_mode_docs:
        for doc_id, actual, expected in sorted(wrong_mode_docs):
            print(f"  - {doc_id}  checkpoint中={actual}  期望={expected}")
    print()

    # ---- 无法在 source 中找到对应文档 ----
    if not_in_expected:
        print(f"[警告] checkpoint 中有 {len(not_in_expected)} 个 doc_id 在 source JSONL 中不存在：")
        for d in sorted(not_in_expected):
            print(f"  - {d}")
        print()

    if unknown_doc_ids:
        print(f"[警告] {len(unknown_doc_ids)} 条 record_id 无法解析 mode 后缀，已跳过：")
        for r in unknown_doc_ids[:10]:
            print(f"  {r}")
        print()

    # ---- 总结 ----
    if not dual_mode_docs and not wrong_mode_docs:
        print("✓ 所有已完成记录的模式分配均与当前期望一致，无异常。")
    else:
        dual_set  = {d for d, _  in dual_mode_docs}
        wrong_set = {d for d, _, _ in wrong_mode_docs}
        print(f"⚠ 问题1（双模式）: {len(dual_set)} 篇")
        print(f"⚠ 问题2（单模式但错误）: {len(wrong_set)} 篇")
        print(f"⚠ 合计需要清理: {len(dual_set | wrong_set)} 篇")
    print("=" * 60)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def main() -> None:
    if not CHECKPOINT_PATH.exists():
        print(f"错误：checkpoint 文件不存在: {CHECKPOINT_PATH}", file=sys.stderr)
        sys.exit(1)
    if not SOURCE_JSONL.exists():
        print(f"错误：source JSONL 不存在: {SOURCE_JSONL}", file=sys.stderr)
        sys.exit(1)

    print(f"加载 checkpoint : {CHECKPOINT_PATH}")
    completed_ids = load_checkpoint(CHECKPOINT_PATH)
    print(f"加载 source JSONL: {SOURCE_JSONL}")
    print(f"random_seed      : {RANDOM_SEED}   (run.py --seed 默认值: 42)")
    print(f"mixing_modes     : {MIXING_MODES}")
    print()
    expected_modes = build_expected_modes(SOURCE_JSONL, RANDOM_SEED, MIXING_MODES)

    analyze(completed_ids, expected_modes)


if __name__ == "__main__":
    main()
