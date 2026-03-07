"""
命令行入口：run.py

用法示例：
  # -------- 批量数据集构建 --------

  # 基本运行（使用 qwen3.5-plus）
  python run.py batch

  # 指定模型
  python run.py batch --model gemini-3.1-flash-lite-preview

  # dry-run 测试（不调用 API，验证完整流程）
  python run.py batch --dry-run --max-docs 10

  # 指定自定义人类文本文件
  python run.py batch --source data/human_texts_10k.jsonl

  # -------- 查看支持的模型 --------
  python run.py list-models

  # -------- 单文本测试 --------
  # 请使用 run_single.py
"""
from __future__ import annotations

import argparse
import asyncio

from src.config import DatasetConfig, SUPPORTED_MODELS
from src.pipeline import DatasetPipeline
from src.utils import get_logger

logger = get_logger("run")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AI 混合数据集构建工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="可用子命令")

    # -------- batch 子命令 --------
    bp = subparsers.add_parser("batch", help="批量构建数据集")
    bp.add_argument(
        "--model", type=str, default="qwen3.5-plus",
        help=f"改写模型 (默认: qwen3.5-plus)",
    )
    bp.add_argument(
        "--source", type=str, default="data/human_texts_1k.jsonl",
        help="预采样人类文本 JSONL 文件路径",
    )
    bp.add_argument(
        "--output-dir", type=str, default="output",
        help="输出目录 (默认: output/)",
    )
    bp.add_argument(
        "--max-docs", type=int, default=None,
        help="最多处理 N 篇文档（调试用）",
    )
    bp.add_argument("--concurrent", type=int, default=8, help="并发请求数")
    bp.add_argument("--seed", type=int, default=42, help="随机种子")
    bp.add_argument("--dry-run", action="store_true", help="不调用 API")

    # -------- list-models 子命令 --------
    subparsers.add_parser("list-models", help="列出所有支持的模型")

    return parser


def list_models() -> None:
    print("\n支持的 LLM 模型：")
    print(f"  {'名称':<40} {'Provider':<12} {'RPM':<8}")
    print("  " + "-" * 62)
    for name, m in SUPPORTED_MODELS.items():
        print(f"  {name:<40} {m.provider:<12} {m.requests_per_minute:<8}")
    print()


# ---------------------------------------------------------------------------
# batch 子命令
# ---------------------------------------------------------------------------

async def run_batch(args: argparse.Namespace) -> None:
    """批量数据集构建。"""
    from pathlib import Path as _Path
    source_tag: str = _Path(args.source).stem

    cfg = DatasetConfig(
        source_path=args.source,
        source_tag=source_tag,
        output_dir=args.output_dir,
        checkpoint_dir=f"{args.output_dir}/checkpoints",
        rewrite_model=args.model,
        concurrent_requests=args.concurrent,
        random_seed=args.seed,
    )

    logger.info("运行配置：")
    logger.info(f"  source_path       = {cfg.source_path}")
    logger.info(f"  rewrite_model     = {cfg.rewrite_model}")
    logger.info(f"  ai_ratios         = {cfg.ai_ratios}")
    logger.info(f"  mixing_modes      = {cfg.mixing_modes}")
    logger.info(f"  concurrent        = {cfg.concurrent_requests}")
    logger.info(f"  dry_run           = {args.dry_run}")

    pipeline = DatasetPipeline(cfg)
    await pipeline.run(max_docs=args.max_docs, dry_run=args.dry_run)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "list-models":
        list_models()
    elif args.command == "batch":
        await run_batch(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
