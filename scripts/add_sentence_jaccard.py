#!/usr/bin/env python3
"""
为现有 JSONL 数据集补充 sentence_jaccard 字段的独立脚本。

定义：
- 仅对 sentence_labels == 1 的句子参与计算
- 每个被 AI 改写句子的分数，使用现有的 Jaccard Distance 定义：
  1 - |A ∩ B| / |A ∪ B|
- 最终 sentence_jaccard 为这些句子分数的平均值

说明：
- 该脚本不修改现有 pipeline，只做离线后处理
- 若 original_text / mixed_text / sentence_labels 三者句子数无法对齐，
  则该条记录的 sentence_jaccard 写为 null，并打印统计

示例：
  python scripts/add_sentence_jaccard.py --input-file output/mixed_dataset.jsonl
  python scripts/add_sentence_jaccard.py --input-file output/mixed_dataset.jsonl --in-place
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sentence_processor import split_into_sentences

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="为 JSONL 数据集补充 sentence_jaccard 字段。",
    )
    parser.add_argument(
        "--input-file",
        required=True,
        help="输入 JSONL 文件路径。",
    )
    parser.add_argument(
        "--output-file",
        default="",
        help="输出 JSONL 文件路径；为空时自动生成。",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="原地更新输入文件。启用后会忽略 --output-file。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已存在的输出文件。",
    )
    return parser.parse_args()


def resolve_output_path(input_path: Path, output_file: str, in_place: bool) -> Path:
    if in_place:
        return input_path
    if output_file:
        return Path(output_file)
    suffix = input_path.suffix or ".jsonl"
    filename = f"{input_path.stem}.sentence_jaccard{suffix}"
    return input_path.with_name(filename)


def _normalize_words(text: str) -> list[str]:
    cleaned = _PUNCT_RE.sub(" ", text.lower())
    return cleaned.split()


def _word_set(text: str) -> set[str]:
    return set(_normalize_words(text))


def compute_jaccard_distance(original_text: str, mixed_text: str) -> float:
    a = _word_set(original_text)
    b = _word_set(mixed_text)
    union = a | b
    if not union:
        return 0.0
    intersection = a & b
    return round(1.0 - (len(intersection) / len(union)), 6)


def compute_sentence_jaccard(record: dict[str, Any]) -> tuple[float | None, str | None]:
    original_text = str(record.get("original_text", "") or "")
    mixed_text = str(record.get("mixed_text", "") or "")
    labels = record.get("sentence_labels", [])

    if not isinstance(labels, list):
        return None, "invalid_sentence_labels"

    ai_indices = [idx for idx, label in enumerate(labels) if int(label) == 1]
    if not ai_indices:
        return 0.0, None

    original_sentences = split_into_sentences(original_text)
    mixed_sentences = split_into_sentences(mixed_text)

    if len(original_sentences) != len(mixed_sentences) or len(original_sentences) != len(labels):
        return None, "sentence_count_mismatch"

    scores = [
        compute_jaccard_distance(original_sentences[idx], mixed_sentences[idx])
        for idx in ai_indices
    ]
    if not scores:
        return 0.0, None
    return round(sum(scores) / len(scores), 6), None


def iter_jsonl(path: Path):
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
            yield line_no, record


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    output_path = resolve_output_path(input_path, args.output_file, args.in_place)
    if output_path.exists() and output_path != input_path and not args.overwrite:
        raise FileExistsError(f"输出文件已存在: {output_path}；如需覆盖请加 --overwrite")

    temp_output_path = (
        input_path.with_suffix(f"{input_path.suffix}.tmp")
        if args.in_place
        else output_path
    )

    total = 0
    zero_ai = 0
    updated = 0
    invalid_labels = 0
    mismatched = 0

    temp_output_path.parent.mkdir(parents=True, exist_ok=True)
    with temp_output_path.open("w", encoding="utf-8") as out_fh:
        for line_no, record in iter_jsonl(input_path):
            total += 1

            value, error_code = compute_sentence_jaccard(record)
            record["sentence_jaccard"] = value

            if error_code is None:
                updated += 1
                if value == 0.0 and sum(int(label) for label in record.get("sentence_labels", [])) == 0:
                    zero_ai += 1
            elif error_code == "invalid_sentence_labels":
                invalid_labels += 1
            elif error_code == "sentence_count_mismatch":
                mismatched += 1
            else:
                raise RuntimeError(f"未处理的错误类型: {error_code} @ line={line_no}")

            out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    if args.in_place:
        temp_output_path.replace(input_path)

    final_output = input_path if args.in_place else output_path
    print(f"已写出: {final_output}")
    print(f"总记录数: {total}")
    print(f"成功写入 sentence_jaccard: {updated}")
    print(f"其中无 AI 句子、写入 0.0: {zero_ai}")
    print(f"sentence_labels 非法: {invalid_labels}")
    print(f"句子数不对齐、写入 null: {mismatched}")


if __name__ == "__main__":
    main()
