"""
Humanize the grouped benchmark test split with the sentence-level rewrite pipeline.

Default input:
  datasplit/benchmark_grouped/test.jsonl

Default output:
  datasplit/benchmark_grouped/test_rewrite.jsonl

Examples:
  python datasplit/rewrite_benchmark_test.py --model MiniMax-M2.7
  python datasplit/rewrite_benchmark_test.py --model qwen3.5-flash --dry-run --max-records 10
"""
from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DatasetConfig  # noqa: E402
from src.rewrite_pipeline import RewriteDatasetPipeline  # noqa: E402
from src.utils import get_logger  # noqa: E402


logger = get_logger("rewrite_benchmark_test")


DEFAULT_INPUT = PROJECT_ROOT / "datasplit" / "benchmark_grouped" / "test.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "datasplit" / "benchmark_grouped" / "test_rewrite.jsonl"


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _materialize_unique_input(input_path: Path) -> tuple[Path, dict[str, str]]:
    rows = _load_rows(input_path)
    tmp_dir = Path(tempfile.mkdtemp(prefix="benchmark_rewrite_", dir=str(PROJECT_ROOT / "datasplit")))
    temp_input = tmp_dir / "test.unique_ids.jsonl"
    id_map: dict[str, str] = {}

    with temp_input.open("w", encoding="utf-8") as fh:
        for line_no, row in enumerate(rows, start=1):
            original_id = str(row.get("id", f"row_{line_no}"))
            temp_id = f"{original_id}__row{line_no:05d}"
            row["id"] = temp_id
            row["_original_id"] = original_id
            id_map[temp_id] = original_id
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    return temp_input, id_map


def _restore_original_ids(output_path: Path, id_map: dict[str, str]) -> None:
    restored_path = output_path.with_suffix(output_path.suffix + ".restored")
    with output_path.open(encoding="utf-8") as in_fh, restored_path.open("w", encoding="utf-8") as out_fh:
        for raw_line in in_fh:
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            temp_id = str(row.get("id", ""))
            if temp_id in id_map:
                row["id"] = id_map[temp_id]
            row.pop("_original_id", None)
            out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    restored_path.replace(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Humanize mixed_text in datasplit/benchmark_grouped/test.jsonl.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="MiniMax-M2.7",
        help="Humanizer model to call. Must exist in src.config.SUPPORTED_MODELS.",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input JSONL path.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--source-model",
        type=str,
        default="benchmark_grouped",
        help="Run-level source_model value written to rewrite metadata.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="English",
        help="Language hint for the humanizer prompt.",
    )
    parser.add_argument(
        "--concurrent-requests",
        type=int,
        default=8,
        help="Maximum concurrent API requests.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Only process the first N records, useful for testing.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override the model default temperature.",
    )
    parser.add_argument(
        "--rpm",
        type=int,
        default=None,
        help="Override the model default requests_per_minute.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Override the model default max_output_tokens.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call APIs; copy selected sentences through unchanged.",
    )
    return parser


async def main() -> None:
    args = build_parser().parse_args()
    input_path = args.input_file.resolve()
    output_path = args.output_file.resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")
    if input_path == output_path:
        raise ValueError("Output file must be different from input file.")

    temp_input_path, id_map = _materialize_unique_input(input_path)
    temp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")

    cfg = DatasetConfig(
        rewrite_model=args.model,
        source_path=str(temp_input_path),
        output_dir=str(temp_output_path.parent),
        concurrent_requests=max(1, args.concurrent_requests),
        temperature=args.temperature,
        requests_per_minute=args.rpm,
        max_output_tokens=args.max_output_tokens,
    )
    model_cfg = cfg.get_model_config()

    logger.info("Benchmark test rewrite config:")
    logger.info("  input_path        = %s", input_path)
    logger.info("  output_path       = %s", output_path)
    logger.info("  source_model      = %s", args.source_model)
    logger.info("  rewrite_model     = %s", cfg.rewrite_model)
    logger.info("  provider          = %s", model_cfg.provider)
    logger.info("  model_id          = %s", model_cfg.model_id)
    logger.info("  concurrent        = %s", cfg.concurrent_requests)
    logger.info("  max_records       = %s", args.max_records)
    logger.info("  dry_run           = %s", args.dry_run)

    pipeline = RewriteDatasetPipeline(
        cfg,
        input_path=temp_input_path,
        output_path=temp_output_path,
        source_model=args.source_model,
        language_hint=args.language,
    )
    try:
        await pipeline.run(max_records=args.max_records, dry_run=args.dry_run)
        _restore_original_ids(temp_output_path, id_map)
        temp_output_path.replace(output_path)
    finally:
        if temp_input_path.exists():
            temp_input_path.unlink()
        if temp_output_path.exists():
            temp_output_path.unlink()
        temp_dir = temp_input_path.parent
        if temp_dir.exists():
            try:
                temp_dir.rmdir()
            except OSError:
                pass


if __name__ == "__main__":
    asyncio.run(main())
