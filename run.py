"""
命令行入口：run.py

用法示例：
  # -------- 批量数据集构建 --------

  # 默认运行
  python run.py batch

  # 指定模型
  python run.py batch --model qwen3.6-plus-preview-free
  python run.py batch --model gemini-3.1-flash-lite-preview
  python run.py batch --model MiniMax-M2.7

  # dry-run 测试（不调用 API，验证完整流程）
  python run.py batch --dry-run --max-docs 10

  # -------- 查看支持的模型 --------
  python run.py list-models

  # -------- 单文本测试 --------
  # 请使用 scripts/run_single.py
"""
from __future__ import annotations

import argparse
import asyncio

from src.config import DatasetConfig, SUPPORTED_MODELS
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
        "--model", type=str, default="MiniMax-M2.7",
        help="改写模型 (默认: MiniMax-M2.7)",
    )
    bp.add_argument(
        "--max-docs", type=int, default=None,
        help="最多处理 N 篇文档（调试用）",
    )
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
    from src.pipeline import DatasetPipeline

    cfg = DatasetConfig(
        rewrite_model=args.model,
    )

    logger.info("运行配置：")
    logger.info(f"  source_path       = {cfg.source_path}")
    logger.info(f"  rewrite_model     = {cfg.rewrite_model}")
    logger.info(f"  ai_ratios         = {cfg.ai_ratios}")
    logger.info(f"  mixing_modes      = {cfg.mixing_modes}")
    logger.info(f"  concurrent        = {cfg.concurrent_requests}")
    logger.info(f"  random_seed       = {cfg.random_seed}")
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
