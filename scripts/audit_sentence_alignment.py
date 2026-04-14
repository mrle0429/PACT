#!/usr/bin/env python3
"""
审计 mixed dataset 的句子级对齐问题，并输出摘要报告。

核心检查：
- original_text / mixed_text / sentence_labels 的句子数是否一致
- mixed_text 的句子数相对 labels 增减了多少
- 是否能从 api_logs 中证明某个“单句改写”被模型输出成了多句

输出：
- report.md   : 人类可读摘要
- summary.json: 结构化统计

示例：
  /Volumes/Mac/Project/PACT/.venv/bin/python scripts/audit_sentence_alignment.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sentence_processor import split_into_sentences
from src.utils import extract_json_from_llm_response

OUTPUT_DIR = PROJECT_ROOT / "output" / "with_sentence_jaccard"
API_LOG_DIR = PROJECT_ROOT / "output" / "api_logs"
REPORT_DIR = PROJECT_ROOT / "analysis" / "sentence_alignment_audit"


def infer_model_name(dataset_path: Path) -> str:
    prefix = "mixed_dataset_"
    suffix = ".jsonl"
    name = dataset_path.name
    if not (name.startswith(prefix) and name.endswith(suffix)):
        raise ValueError(f"无法从文件名推断模型名: {dataset_path}")
    return name[len(prefix):-len(suffix)]


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def truncate(text: str, limit: int = 220) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def load_log_profiles(model_name: str) -> dict[str, dict[str, Any]]:
    """
    返回:
      task_id -> {
        "rewrite_sentence_counts": {"3": 1, "4": 2, ...},
        "raw_response": "...",
      }
    """
    profiles: dict[str, dict[str, Any]] = {}
    for log_path in sorted(API_LOG_DIR.glob(f"{model_name}*.jsonl")):
        for record in iter_jsonl(log_path):
            task_id = record.get("task_id")
            raw_response = str(record.get("raw_response", "") or "")
            if not task_id or not raw_response:
                continue
            try:
                parsed = extract_json_from_llm_response(raw_response)
            except Exception:
                continue

            rewrite_sentence_counts: dict[str, int] = {}
            for key, value in parsed.items():
                if isinstance(value, str):
                    rewrite_sentence_counts[str(key)] = len(split_into_sentences(value))

            profiles[str(task_id)] = {
                "rewrite_sentence_counts": rewrite_sentence_counts,
                "raw_response": raw_response,
            }
    return profiles


def classify_reason(
    *,
    mixed_count: int,
    label_count: int,
    rewrite_sentence_counts: dict[str, int] | None,
) -> str:
    if rewrite_sentence_counts is None:
        return "missing_api_log"
    if any(count > 1 for count in rewrite_sentence_counts.values()):
        return "rewrite_split_into_multiple_sentences"
    if mixed_count < label_count:
        return "boundary_merge_or_lost_sentence_boundary"
    if mixed_count > label_count:
        return "boundary_split_or_quote_punctuation_shift"
    return "unknown_alignment_failure"


def build_example(
    record: dict[str, Any],
    *,
    original_sentences: list[str],
    mixed_sentences: list[str],
    rewrite_sentence_counts: dict[str, int] | None,
) -> dict[str, Any]:
    labels = list(record.get("sentence_labels", []))
    ai_indices = [idx for idx, label in enumerate(labels) if int(label) == 1]

    example: dict[str, Any] = {
        "record_id": record.get("id"),
        "counts": {
            "original_sentences": len(original_sentences),
            "mixed_sentences": len(mixed_sentences),
            "sentence_labels": len(labels),
        },
        "ai_indices": ai_indices,
        "rewrite_sentence_counts": rewrite_sentence_counts,
    }

    if ai_indices:
        idx = ai_indices[min(len(ai_indices) - 1, 0)]
        if idx < len(original_sentences):
            example["first_ai_original_sentence"] = truncate(original_sentences[idx])
        if idx < len(mixed_sentences):
            example["first_ai_mixed_sentence"] = truncate(mixed_sentences[idx])

    example["mixed_tail"] = [truncate(sentence, 140) for sentence in mixed_sentences[-3:]]
    return example


def audit_dataset(dataset_path: Path) -> dict[str, Any]:
    model_name = infer_model_name(dataset_path)
    log_profiles = load_log_profiles(model_name)

    total_records = 0
    null_records = 0
    delta_counter: Counter[int] = Counter()
    reason_counter: Counter[str] = Counter()
    examples: dict[str, dict[str, Any]] = {}

    for record in iter_jsonl(dataset_path):
        total_records += 1
        if record.get("sentence_jaccard") is not None:
            continue

        null_records += 1
        labels = list(record.get("sentence_labels", []))
        original_sentences = split_into_sentences(str(record.get("original_text", "") or ""))
        mixed_sentences = split_into_sentences(str(record.get("mixed_text", "") or ""))

        delta = len(mixed_sentences) - len(labels)
        delta_counter[delta] += 1

        profile = log_profiles.get(str(record.get("id")))
        rewrite_sentence_counts = None if profile is None else profile["rewrite_sentence_counts"]
        reason = classify_reason(
            mixed_count=len(mixed_sentences),
            label_count=len(labels),
            rewrite_sentence_counts=rewrite_sentence_counts,
        )
        reason_counter[reason] += 1

        if reason not in examples:
            examples[reason] = build_example(
                record,
                original_sentences=original_sentences,
                mixed_sentences=mixed_sentences,
                rewrite_sentence_counts=rewrite_sentence_counts,
            )

    return {
        "dataset_file": dataset_path.name,
        "model_name": model_name,
        "total_records": total_records,
        "null_sentence_jaccard_records": null_records,
        "null_ratio": round((null_records / total_records), 6) if total_records else 0.0,
        "mixed_minus_label_count_distribution": dict(sorted(delta_counter.items(), key=lambda item: item[0])),
        "reason_distribution": dict(reason_counter),
        "examples": examples,
    }


def render_report(results: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# Sentence Alignment Audit")
    lines.append("")
    lines.append("`sentence_jaccard = null` 表示该记录在句子级无法安全对齐。")
    lines.append("")

    for result in results:
        lines.append(f"## {result['dataset_file']}")
        lines.append("")
        lines.append(f"- model: `{result['model_name']}`")
        lines.append(f"- total_records: {result['total_records']}")
        lines.append(f"- null_sentence_jaccard_records: {result['null_sentence_jaccard_records']}")
        lines.append(f"- null_ratio: {result['null_ratio']:.4%}")
        lines.append(f"- mixed_minus_label_count_distribution: `{result['mixed_minus_label_count_distribution']}`")
        lines.append(f"- reason_distribution: `{result['reason_distribution']}`")
        lines.append("")

        for reason, example in result["examples"].items():
            lines.append(f"### Example: {reason}")
            lines.append("")
            lines.append(f"- record_id: `{example['record_id']}`")
            lines.append(f"- counts: `{example['counts']}`")
            lines.append(f"- ai_indices: `{example['ai_indices']}`")
            lines.append(f"- rewrite_sentence_counts: `{example['rewrite_sentence_counts']}`")
            if "first_ai_original_sentence" in example:
                lines.append(f"- first_ai_original_sentence: `{example['first_ai_original_sentence']}`")
            if "first_ai_mixed_sentence" in example:
                lines.append(f"- first_ai_mixed_sentence: `{example['first_ai_mixed_sentence']}`")
            lines.append(f"- mixed_tail: `{example['mixed_tail']}`")
            lines.append("")

    return "\n".join(lines)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_paths = sorted(OUTPUT_DIR.glob("mixed_dataset_*.jsonl"))
    results = [audit_dataset(path) for path in dataset_paths]

    summary_path = REPORT_DIR / "summary.json"
    report_path = REPORT_DIR / "report.md"

    with summary_path.open("w", encoding="utf-8") as file_obj:
        json.dump(results, file_obj, ensure_ascii=False, indent=2)

    report_path.write_text(render_report(results), encoding="utf-8")

    print(f"summary -> {summary_path}")
    print(f"report  -> {report_path}")


if __name__ == "__main__":
    main()
