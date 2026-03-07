#!/usr/bin/env python
"""
单文本测试工具 — 直接修改下方参数即可运行

用法：
  1. 修改 "用户参数" 区域的变量
  2. 运行: python run_single.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_writer import JsonlWriter
from src.pipeline import process_single_text
from src.utils import get_logger

logger = get_logger("single_test")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                        ✏️  用户参数 — 在这里修改                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# 输入文本（直接粘贴）
TEXT = """
Kane was speaking as he accepted the player of the NIFWA month award which recognised his team's League Cup win and top-six finish in the Premiership. Penalty and free-kick specialist Kane has scored 18 goals this season. \"From fearing relegation last season we have made great progress under David Jeffrey and his assistant Bryan McLoughlin,\" said 29-year-old Kane. \"They have been able to get every ounce out of the players. \"They are great motivators and I'm delighted because this season things have really clicked for me. \"My game has moved up a level and I feel more valuable to the team. \"People say if you score goals you will get recognised and that's the case. But I just want to win games of football, I don't care who scores. \"Our objective at the start of the season was to make the top six and win a trophy and we have managed to do that.\" Ballymena won the League Cup by beating Carrick Rangers 2-0 in the final at Seaview. Last Saturday, though, they crashed 4-0 at home to derby rivals Coleraine in the Irish Cup quarter-finals. The player of the month awards are organised by the Northern Ireland Football Writers' Association.""".strip()

# AI 浓度：单个值 或 列表批量对比
#   单个: RATIOS = [0.4]
#   批量: RATIOS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
RATIOS = [1.0]

# 混合模式：单个 或 列表
#   单个: MODES = ["block_replace"]
#   批量: MODES = ["block_replace", "random_scatter"]
MODES = ["random_scatter"]

# 改写模型
# 可选（见 src/config.py: SUPPORTED_MODELS）:
#       qwen3.5-plus, MiniMax-M2.5, gemini-3.1-flash-lite-preview
MODEL = "qwen3.5-flash"

# dry-run 模式：True = 不调 API，用原句占位（测试流程）
DRY_RUN = False

# 语言提示
LANGUAGE = "English"

# 随机种子
SEED = 42

# 输出文件路径（设为 None 则不保存，仅打印）
OUTPUT_FILE = "output/single_test.jsonl"

# ═══════════════════════════════════════════════════════════════════════════
#                         以下无需修改
# ═══════════════════════════════════════════════════════════════════════════


async def run(
    text: str = TEXT,
    ratios: list[float] | None = None,
    modes: list[str] | None = None,
    model: str = MODEL,
    dry_run: bool = DRY_RUN,
    language: str = LANGUAGE,
    seed: int = SEED,
    output_file: str | None = OUTPUT_FILE,
) -> None:
    """
    执行单文本混合改写测试。

    也可在其他脚本中 import 调用：
        from run_single import run
        import asyncio
        asyncio.run(run(text="...", ratios=[0.4], dry_run=False))
    """
    if ratios is None:
        ratios = RATIOS
    if modes is None:
        modes = MODES

    if not text or not text.strip():
        print("❌ 输入文本为空，退出。")
        return

    # 生成 ratio × mode 组合（0.0 和 1.0 两种模式等价，去重）
    combinations = []
    for ratio in ratios:
        for mode in modes:
            if (ratio == 0.0 or ratio == 1.0) and mode != modes[0]:
                continue
            combinations.append((ratio, mode))

    logger.info(f"待测试组合: {len(combinations)} 个 | model={model} | dry_run={dry_run}")
    logger.info(f"输入文本: {text[:80]}{'...' if len(text) > 80 else ''}")

    # 准备输出
    writer = None
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = JsonlWriter(output_path)

    try:
        for i, (ratio, mode) in enumerate(combinations, 1):
            logger.info(f"[{i}/{len(combinations)}] ratio={ratio:.0%} mode={mode}")

            result = await process_single_text(
                text=text,
                target_ratio=ratio,
                mixing_mode=mode,
                model=model,
                seed=seed,
                dry_run=dry_run,
                language_hint=language,
            )

            # 打印摘要
            print(result.summary())
            print()

            # 转为标准 DatasetRecord 并保存
            if writer is not None:
                record = result.to_dataset_record()
                writer.write(record)

    finally:
        if writer is not None:
            writer.close()
            print(f"\n✅ 结果已保存到: {output_file}")


if __name__ == "__main__":
    asyncio.run(run())
