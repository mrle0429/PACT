#!/usr/bin/env python3
"""
绘制混合数据集的比例一致性分析图。

用法：
    python scripts/plot_ratio_analysis.py
    python scripts/plot_ratio_analysis.py --input output/mixed_dataset_llama4-fast:latest.jsonl
    python scripts/plot_ratio_analysis.py --output analysis/ratio_analysis.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "output" / "mixed_dataset_llama4-fast:latest.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "analysis" / "ratio_analysis.png"


def analyze_dataset_distributions(jsonl_path: Path, output_path: Path) -> None:
    ratios: list[float] = []
    lirs: list[float] = []
    jaccards: list[float | None] = []

    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            data = json.loads(line.strip())
            ratios.append(data["target_ai_ratio"])
            lirs.append(data["lir"])
            jaccards.append(data.get("jaccard_distance"))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.regplot(x=ratios, y=lirs, scatter_kws={"alpha": 0.3}, line_kws={"color": "red"})
    plt.plot([0, 1], [0, 1], "k--", label="y=x (Ideal)")
    plt.title("Target AI Ratio vs LIR")
    plt.xlabel("Set Target AI Ratio")
    plt.ylabel("Actual LIR")
    plt.legend()

    valid_jaccards = [(ratio, value) for ratio, value in zip(ratios, jaccards) if value is not None]
    if valid_jaccards:
        x_vals, y_vals = zip(*valid_jaccards)
        plt.subplot(1, 2, 2)
        sns.regplot(x=list(x_vals), y=list(y_vals), scatter_kws={"alpha": 0.3}, line_kws={"color": "red"})
        plt.title("Target Ratio vs Jaccard Distance")
        plt.xlabel("Set Target AI Ratio")
        plt.ylabel("Jaccard Distance")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"图表已保存为 {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制混合数据集比例分析图")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="输入 JSONL 路径")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出图片路径")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"输入文件不存在: {args.input}")

    analyze_dataset_distributions(args.input, args.output)


if __name__ == "__main__":
    main()
