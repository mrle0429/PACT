#!/usr/bin/env python3
"""
已采样人类文本清洗入口脚本。

用法：
    python scripts/clean_human_texts.py \
        --input data/human_texts_1k.jsonl \
        --output data/human_texts_1k.cleaned.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="清洗已采样的人类文本 JSONL")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入 JSONL 路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出 JSONL 路径",
    )
    parser.add_argument(
        "--min-sentences-to-keep",
        type=int,
        default=1,
        help="清洗后保留样本所需的最少句子数，默认 1",
    )
    parser.add_argument(
        "--use-ftfy",
        action="store_true",
        help="启用更激进的 ftfy 文本修复；默认关闭以保留原始书写表面形式",
    )
    parser.add_argument(
        "--log-output",
        type=str,
        default=None,
        help="逐条清洗日志 JSONL 路径，默认基于输出路径自动生成",
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default=None,
        help="清洗摘要 JSON 路径，默认基于输出路径自动生成",
    )
    args = parser.parse_args()

    from src.preprocess import HumanTextCleaner, HumanTextCleanerConfig

    log_output = args.log_output or _derive_companion_path(args.output, ".cleaning_log.jsonl")
    summary_output = args.summary_output or _derive_companion_path(args.output, ".cleaning_summary.json")
    cleaner = HumanTextCleaner(HumanTextCleanerConfig(
        use_ftfy_if_available=args.use_ftfy,
        min_sentences_to_keep=args.min_sentences_to_keep,
    ))
    stats = cleaner.clean_dataset(
        args.input,
        args.output,
        log_path=log_output,
        summary_path=summary_output,
    )

    summary = {
        "total_records": stats.total_records,
        "written_records": stats.written_records,
        "skipped_records": stats.skipped_records,
        "changed_records": stats.changed_records,
        "total_removed_sentences": stats.total_removed_sentences,
        "logged_records": stats.logged_records,
        "removal_reasons": dict(stats.removal_reasons),
        "log_output": log_output,
        "summary_output": summary_output,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _derive_companion_path(output_path: str, suffix: str) -> str:
    path = output_path
    if path.endswith(".jsonl"):
        return path[:-6] + suffix
    return path + suffix


if __name__ == "__main__":
    main()
