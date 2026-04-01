"""
数据集质量检查脚本

检查项目：
1. JSON 解析：行是否可正常解析
2. 必填字段：id / text / sentence_count 是否存在
3. 字段类型：id 为字符串、text 为字符串、sentence_count 为整数
4. 空文本：text 是否为空或纯空白
5. sentence_count 准确性：与 pysbd 实际分句结果是否一致
6. 空句检测：分句后是否含有空串或纯空白句子
7. 重复 ID：是否存在重复的 id 字段
8. 重复文本：是否存在完全相同的 text
9. 单句文档：sentence_count == 1（可能过短）
10. sentence_count 与 text 长度极端不一致（疑似噪声）
11. 字段 sentence_count 为 0 或负数
12. 来源分布统计（id 前缀分析）
13. 文本长度分布（字符数 / 词数）
14. 异常短句（< 10 字符）在各文档中的数量
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pysbd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ──────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────
DEFAULT_FILE = PROJECT_ROOT / "data" / "human_texts_1k.cleaned.jsonl"
SENT_COUNT_MISMATCH_TOLERANCE = 0  # 允许误差句数（0 = 完全匹配）
SHORT_SENTENCE_CHARS = 10          # 短句阈值（字符数）
SHORT_TEXT_WORDS = 20              # 极短文本阈值（词数）
LONG_TEXT_WORDS = 2000             # 极长文本阈值（词数）

segmenter = pysbd.Segmenter(language="en", clean=False)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in segmenter.segment(text) if s.strip()]


# ──────────────────────────────────────────────
# 主检查逻辑
# ──────────────────────────────────────────────
def check_file(filepath: Path) -> None:
    print(f"{'='*60}")
    print(f"  数据质量检查报告")
    print(f"  文件: {filepath}")
    print(f"{'='*60}\n")

    lines = filepath.read_text(encoding="utf-8").splitlines()
    total_lines = len(lines)
    print(f"[基本信息] 总行数: {total_lines}\n")

    # 计数器
    parse_errors: list[int] = []
    missing_fields: list[tuple[int, list[str]]] = []
    type_errors: list[tuple[int, str]] = []
    empty_texts: list[int] = []
    sent_count_mismatches: list[tuple[int, str, int, int]] = []  # (lineno, id, stored, actual)
    empty_sentence_records: list[tuple[int, str, list[int]]] = []  # (lineno, id, empty_idx)
    short_sentence_records: list[tuple[int, str, list[tuple[int, str]]]] = []
    zero_or_neg_count: list[tuple[int, str, int]] = []
    single_sentence_docs: list[tuple[int, str]] = []
    short_texts: list[tuple[int, str, int]] = []  # (lineno, id, word_count)
    long_texts: list[tuple[int, str, int]] = []

    id_counter: Counter[str] = Counter()
    text_to_lines: dict[str, list[int]] = defaultdict(list)
    source_counter: Counter[str] = Counter()
    word_counts: list[int] = []
    char_counts: list[int] = []
    sentence_counts_actual: list[int] = []

    for lineno, raw in enumerate(lines, start=1):
        raw = raw.strip()
        if not raw:
            continue

        # ① JSON 解析
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as e:
            parse_errors.append(lineno)
            continue

        # ② 必填字段
        missing = [f for f in ("id", "text", "sentence_count") if f not in record]
        if missing:
            missing_fields.append((lineno, missing))
            continue

        rid = record["id"]
        text = record["text"]
        sc = record["sentence_count"]

        # ③ 字段类型
        if not isinstance(rid, str):
            type_errors.append((lineno, f"id 类型错误: {type(rid).__name__}"))
        if not isinstance(text, str):
            type_errors.append((lineno, f"text 类型错误: {type(text).__name__}"))
            continue  # text 异常时跳过后续依赖 text 的检查
        if not isinstance(sc, int):
            type_errors.append((lineno, f"sentence_count 类型错误: {type(sc).__name__} (值={sc!r})"))

        # ④ 空文本
        if not text.strip():
            empty_texts.append(lineno)
            continue

        # ⑤ sentence_count 为 0 或负
        if isinstance(sc, int) and sc <= 0:
            zero_or_neg_count.append((lineno, rid, sc))

        # ⑥ 重复 ID
        id_counter[rid] += 1

        # ⑦ 重复文本
        text_to_lines[text].append(lineno)

        # ⑧ 来源统计（取 id 中 _ 前的前缀）
        prefix = rid.split("_")[0] if "_" in rid else rid[:8]
        source_counter[prefix] += 1

        # ⑨ 文本长度
        words = text.split()
        wc = len(words)
        cc = len(text)
        word_counts.append(wc)
        char_counts.append(cc)
        if wc < SHORT_TEXT_WORDS:
            short_texts.append((lineno, rid, wc))
        if wc > LONG_TEXT_WORDS:
            long_texts.append((lineno, rid, wc))

        # ⑩ 分句与 sentence_count 校验
        sentences = split_sentences(text)
        actual_sc = len(sentences)
        sentence_counts_actual.append(actual_sc)

        if isinstance(sc, int):
            diff = abs(sc - actual_sc)
            if diff > SENT_COUNT_MISMATCH_TOLERANCE:
                sent_count_mismatches.append((lineno, rid, sc, actual_sc))

        # ⑪ 单句文档
        if actual_sc == 1:
            single_sentence_docs.append((lineno, rid))

        # ⑫ 空句检测（理论上 split_sentences 已过滤，但双重检查原始分句）
        raw_sentences = segmenter.segment(text)
        empty_indices = [i for i, s in enumerate(raw_sentences) if not s.strip()]
        if empty_indices:
            empty_sentence_records.append((lineno, rid, empty_indices))

        # ⑬ 异常短句
        short_sents = [(i, s) for i, s in enumerate(sentences) if len(s) < SHORT_SENTENCE_CHARS]
        if short_sents:
            short_sentence_records.append((lineno, rid, short_sents))

    # ──────────────────────────────────────────────
    # 输出报告
    # ──────────────────────────────────────────────

    def section(title: str) -> None:
        print(f"\n{'─'*50}")
        print(f"  {title}")
        print(f"{'─'*50}")

    # ── 1. JSON 解析错误
    section("① JSON 解析错误")
    if parse_errors:
        print(f"  [FAIL] {len(parse_errors)} 行解析失败: 行号 {parse_errors}")
    else:
        print("  [OK] 全部行均可正常解析")

    # ── 2. 缺失字段
    section("② 必填字段缺失")
    if missing_fields:
        print(f"  [FAIL] {len(missing_fields)} 条记录缺少字段:")
        for lineno, fields in missing_fields[:20]:
            print(f"    第 {lineno} 行 — 缺少: {fields}")
    else:
        print("  [OK] 所有记录均包含 id / text / sentence_count")

    # ── 3. 字段类型错误
    section("③ 字段类型错误")
    if type_errors:
        print(f"  [FAIL] {len(type_errors)} 处类型异常:")
        for lineno, msg in type_errors[:20]:
            print(f"    第 {lineno} 行 — {msg}")
    else:
        print("  [OK] 所有字段类型正常")

    # ── 4. 空文本
    section("④ 空文本 (text 为空或纯空白)")
    if empty_texts:
        print(f"  [FAIL] {len(empty_texts)} 条: 行号 {empty_texts}")
    else:
        print("  [OK] 无空文本")

    # ── 5. sentence_count 为 0 或负
    section("⑤ sentence_count 为 0 或负数")
    if zero_or_neg_count:
        print(f"  [FAIL] {len(zero_or_neg_count)} 条:")
        for lineno, rid, sc in zero_or_neg_count[:20]:
            print(f"    第 {lineno} 行 id={rid!r} sentence_count={sc}")
    else:
        print("  [OK] 无异常值")

    # ── 6. sentence_count 与实际分句不一致
    section(f"⑥ sentence_count 与 pysbd 实际分句不一致 (允许误差={SENT_COUNT_MISMATCH_TOLERANCE})")
    if sent_count_mismatches:
        print(f"  [WARN] {len(sent_count_mismatches)} 条不一致 (共 {total_lines} 条):")
        for lineno, rid, stored, actual in sent_count_mismatches[:30]:
            diff = actual - stored
            print(f"    第 {lineno:>4} 行  id={rid!r:<30}  stored={stored:>3}  actual={actual:>3}  diff={diff:+d}")
        if len(sent_count_mismatches) > 30:
            print(f"    ... 共 {len(sent_count_mismatches)} 条，仅显示前 30 条")
    else:
        print("  [OK] 全部一致")

    # ── 7. 空句
    section("⑦ 空句 (pysbd 原始分句后含有空串)")
    if empty_sentence_records:
        print(f"  [WARN] {len(empty_sentence_records)} 条记录含空句:")
        for lineno, rid, idxs in empty_sentence_records[:20]:
            print(f"    第 {lineno} 行 id={rid!r}  空句位置索引: {idxs}")
    else:
        print("  [OK] 无空句")

    # ── 8. 异常短句
    section(f"⑧ 异常短句 (< {SHORT_SENTENCE_CHARS} 字符)")
    if short_sentence_records:
        print(f"  [INFO] {len(short_sentence_records)} 条记录含异常短句:")
        for lineno, rid, shorts in short_sentence_records[:20]:
            for idx, s in shorts[:3]:
                print(f"    第 {lineno:>4} 行 id={rid!r}  句[{idx}]: {s!r}")
        if len(short_sentence_records) > 20:
            print(f"    ... 共 {len(short_sentence_records)} 条，仅显示前 20 条")
    else:
        print(f"  [OK] 无异常短句")

    # ── 9. 重复 ID
    section("⑨ 重复 ID")
    dup_ids = {k: v for k, v in id_counter.items() if v > 1}
    if dup_ids:
        print(f"  [FAIL] {len(dup_ids)} 个 ID 重复:")
        for rid, cnt in list(dup_ids.items())[:20]:
            print(f"    id={rid!r}  出现 {cnt} 次")
    else:
        print("  [OK] 无重复 ID")

    # ── 10. 重复文本
    section("⑩ 重复文本")
    dup_texts = {t: ls for t, ls in text_to_lines.items() if len(ls) > 1}
    if dup_texts:
        print(f"  [FAIL] {len(dup_texts)} 段文本重复:")
        for text, ls in list(dup_texts.items())[:10]:
            preview = text[:60].replace("\n", " ")
            print(f"    行号 {ls}  内容前缀: {preview!r}")
    else:
        print("  [OK] 无重复文本")

    # ── 11. 单句文档
    section("⑪ 单句文档 (sentence_count_actual == 1)")
    if single_sentence_docs:
        print(f"  [INFO] {len(single_sentence_docs)} 条仅含 1 句:")
        for lineno, rid in single_sentence_docs[:20]:
            print(f"    第 {lineno} 行 id={rid!r}")
    else:
        print("  [OK] 无单句文档")

    # ── 12. 极短/极长文本
    section(f"⑫ 极短文本 (< {SHORT_TEXT_WORDS} 词) / 极长文本 (> {LONG_TEXT_WORDS} 词)")
    if short_texts:
        print(f"  [INFO] {len(short_texts)} 条极短文本:")
        for lineno, rid, wc in short_texts[:10]:
            print(f"    第 {lineno} 行 id={rid!r}  {wc} 词")
    else:
        print(f"  [OK] 无极短文本")
    if long_texts:
        print(f"  [INFO] {len(long_texts)} 条极长文本:")
        for lineno, rid, wc in long_texts[:10]:
            print(f"    第 {lineno} 行 id={rid!r}  {wc} 词")
    else:
        print(f"  [OK] 无极长文本")

    # ── 13. 来源分布
    section("⑬ 来源分布 (id 前缀统计)")
    total_valid = sum(source_counter.values())
    for src, cnt in sorted(source_counter.items(), key=lambda x: -x[1]):
        pct = cnt / total_valid * 100 if total_valid else 0
        print(f"  {src:<12}  {cnt:>5} 条  ({pct:.1f}%)")

    # ── 14. 文本长度分布
    section("⑭ 文本长度分布")
    if word_counts:
        import statistics
        wc_sorted = sorted(word_counts)
        n = len(wc_sorted)
        print(f"  词数  — min={min(wc_sorted)}  max={max(wc_sorted)}  "
              f"mean={statistics.mean(wc_sorted):.1f}  "
              f"median={statistics.median(wc_sorted):.0f}  "
              f"p5={wc_sorted[max(0,int(n*0.05))]:.0f}  "
              f"p95={wc_sorted[min(n-1,int(n*0.95))]:.0f}")
        cc_sorted = sorted(char_counts)
        print(f"  字符数 — min={min(cc_sorted)}  max={max(cc_sorted)}  "
              f"mean={statistics.mean(cc_sorted):.1f}  "
              f"median={statistics.median(cc_sorted):.0f}")
        sc_sorted = sorted(sentence_counts_actual)
        print(f"  句数   — min={min(sc_sorted)}  max={max(sc_sorted)}  "
              f"mean={statistics.mean(sc_sorted):.1f}  "
              f"median={statistics.median(sc_sorted):.0f}  "
              f"p5={sc_sorted[max(0,int(n*0.05))]:.0f}  "
              f"p95={sc_sorted[min(n-1,int(n*0.95))]:.0f}")

    # ── 汇总
    section("汇总")
    issues = sum([
        len(parse_errors),
        len(missing_fields),
        len(type_errors),
        len(empty_texts),
        len(zero_or_neg_count),
        len(sent_count_mismatches),
        len(empty_sentence_records),
        len(dup_ids),
        len(dup_texts),
    ])
    warnings = sum([
        len(short_sentence_records),
        len(single_sentence_docs),
        len(short_texts),
        len(long_texts),
    ])
    print(f"  严重问题 (FAIL/WARN): {issues} 项")
    print(f"  信息提示 (INFO):       {warnings} 项")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    filepath = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FILE
    if not filepath.exists():
        print(f"[ERROR] 文件不存在: {filepath}")
        sys.exit(1)
    check_file(filepath)
