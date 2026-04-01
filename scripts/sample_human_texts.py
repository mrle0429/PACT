#!/usr/bin/env python3
"""
人类文本采样入口脚本 — 从 4 个 HuggingFace 数据集各采样 2,50 条，
生成 10,00 条中间态人类文本数据集。

用法：
    python scripts/sample_human_texts.py                           # 默认参数
    python scripts/sample_human_texts.py --total 1000 --seed 123  # 自定义
    python scripts/sample_human_texts.py --output data/my.jsonl   # 自定义输出路径
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="从四类数据集采样人类文本（中间态数据集）"
    )
    parser.add_argument(
        "--total", type=int, default=10,
        help="总采样数量（将平均分配到 4 个数据源），默认 10",
    )
    parser.add_argument(
        "--seed", type=int, default=2006,
        help="随机种子，默认 2006",
    )
    parser.add_argument(
        "--output", type=str, default="data/human_texts_10.jsonl",
        help="输出文件路径，默认 data/human_texts_10.jsonl",
    )
    parser.add_argument(
        "--min-sentences", type=int, default=8,
        help="最少句子数，默认 8",
    )
    parser.add_argument(
        "--max-sentences", type=int, default=20,
        help="最多句子数，默认 20",
    )
    parser.add_argument(
        "--max-chars", type=int, default=8000,
        help="字符数上限，默认 8000",
    )
    args = parser.parse_args()

    if args.total % 4 != 0:
        print(f"警告: total={args.total} 不是 4 的倍数，各数据源配额会略有差异")

    from src.preprocess import FilterConfig, build_human_dataset

    filter_cfg = FilterConfig(
        min_sentences=args.min_sentences,
        max_sentences=args.max_sentences,
        max_chars=args.max_chars,
        random_seed=args.seed,
    )

    t0 = time.time()
    out_path = build_human_dataset(
        output_path=args.output,
        total=args.total,
        seed=args.seed,
        filter_cfg=filter_cfg,
    )
    elapsed = time.time() - t0

    print(f"\n✅ 完成! 输出: {out_path}  耗时: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
