"""
命令行入口：run.py

用法示例：
  # -------- 批量数据集构建 --------

  # 默认运行
  python run.py batch

  # 指定模型
  python run.py batch --model qwen3.5-flash
  python run.py batch --model qwen3.5-plus
  python run.py batch --model qwen3.6-plus
  python run.py batch --model qwen3.6-plus-preview-free
  python run.py batch --model DeepSeek-V3.2
  python run.py batch --model doubao-seed-2-0-pro
  python run.py batch --model claude-haiku-4.5
  python run.py batch --model gemini-3.1-flash-lite-preview
  python run.py batch --model gemma4
  python run.py batch --model MiniMax-M2.7
  python run.py batch --model gpt-5.4
  python run.py batch --model llama4-fast:latest

  # dry-run 测试（不调用 API，验证完整流程）
  python run.py batch --dry-run --max-docs 10

  # -------- 查看支持的模型 --------
  python run.py list-models

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
        help="改写模型",
    )
    bp.add_argument(
        "--max-docs", type=int, default=None,
        help="最多处理 N 篇文档（调试用）",
    )
    bp.add_argument(
        "--temperature", type=float, default=None,
        help="覆盖模型默认 temperature",
    )
    bp.add_argument(
        "--rpm", type=int, default=None,
        help="覆盖模型默认 requests_per_minute",
    )
    bp.add_argument(
        "--max-output-tokens", type=int, default=None,
        help="覆盖模型默认 max_output_tokens",
    )
    bp.add_argument("--dry-run", action="store_true", help="不调用 API")

    # -------- list-models 子命令 --------
    subparsers.add_parser("list-models", help="列出所有支持的模型")

    return parser


def list_models() -> None:
    print("\n支持的 LLM 模型：")
    print(f"  {'名称':<40} {'Provider':<12} {'Temp':<8} {'RPM':<8} {'MaxTokens':<10}")
    print("  " + "-" * 88)
    for name, m in SUPPORTED_MODELS.items():
        print(
            f"  {name:<40} {m.provider:<12} {m.temperature:<8} "
            f"{m.requests_per_minute:<8} {m.max_output_tokens:<10}"
        )
    print()


# ---------------------------------------------------------------------------
# batch 子命令
# ---------------------------------------------------------------------------

async def run_batch(args: argparse.Namespace) -> None:
    """批量数据集构建。"""
    from src.pipeline import DatasetPipeline

    cfg = DatasetConfig(
        rewrite_model=args.model,
        temperature=args.temperature,
        requests_per_minute=args.rpm,
        max_output_tokens=args.max_output_tokens,
    )
    model_cfg = cfg.get_model_config()

    logger.info("运行配置：")
    logger.info(f"  source_path       = {cfg.source_path}")
    logger.info(f"  rewrite_model     = {cfg.rewrite_model}")
    logger.info(f"  provider          = {model_cfg.provider}")
    logger.info(f"  model_id          = {model_cfg.model_id}")
    logger.info(f"  temperature       = {model_cfg.temperature}")
    logger.info(f"  requests_per_min  = {model_cfg.requests_per_minute}")
    logger.info(f"  max_output_tokens = {model_cfg.max_output_tokens}")
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
