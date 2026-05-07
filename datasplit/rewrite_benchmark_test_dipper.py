"""
Rewrite `datasplit/benchmark_grouped/test.jsonl` with a sentence-level DIPPER HTTP service.

This script is intentionally standalone:
- it does not touch the main rewrite pipeline
- it does not use id-based deduplication
- it rewrites only sentences with `sentence_labels == 1`
- it keeps the existing strict validation rules

Default service:
  http://127.0.0.1:6006/paraphrase
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DatasetConfig  # noqa: E402
from src.label_calculator import compute_labels  # noqa: E402
from src.sentence_processor import split_into_sentences  # noqa: E402
from src.utils import get_logger  # noqa: E402


logger = get_logger("rewrite_benchmark_test_dipper")


DEFAULT_INPUT = PROJECT_ROOT / "datasplit" / "benchmark_grouped" / "test.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "datasplit" / "benchmark_grouped" / "test_rewrite_dipper.jsonl"


@dataclass(frozen=True)
class DipperConfig:
    base_url: str = "http://127.0.0.1:6006"
    lex: int = 40
    order: int = 100
    max_new_tokens: int = 80
    timeout_seconds: float = 120.0
    concurrent_requests: int = 4
    retries: int = 2


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _normalize_labels(value: Any, task_id: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"[{task_id}] sentence_labels 必须是 list")
    labels: list[int] = []
    for index, item in enumerate(value):
        label = int(item)
        if label not in {0, 1}:
            raise ValueError(f"[{task_id}] sentence_labels[{index}] 必须为 0/1")
        labels.append(label)
    return labels


def _build_prompt_meta(base_url: str, lex: int, order: int, max_new_tokens: int) -> dict[str, Any]:
    return {
        "rewriter": "dipper-http",
        "endpoint": f"{base_url.rstrip('/')}/paraphrase",
        "lex": lex,
        "order": order,
        "max_new_tokens": max_new_tokens,
    }


async def _call_dipper(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    *,
    text: str,
    cfg: DipperConfig,
) -> str:
    payload = {
        "text": text,
        "lex": cfg.lex,
        "order": cfg.order,
        "max_new_tokens": cfg.max_new_tokens,
    }

    last_error: Exception | None = None
    for attempt in range(1, cfg.retries + 2):
        try:
            async with sem:
                response = await client.post("/paraphrase", json=payload)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict):
                candidate = data.get("output", data.get("text", ""))
            else:
                candidate = data
            if not isinstance(candidate, str):
                raise ValueError(f"unexpected response payload: {type(candidate).__name__}")
            candidate = candidate.strip()
            if not candidate:
                raise ValueError("empty dipper output")
            return candidate
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt > cfg.retries:
                break
            await asyncio.sleep(min(2.0 * attempt, 5.0))

    raise RuntimeError(f"DIPPER request failed: {last_error}") from last_error


async def _rewrite_row(
    row: dict[str, Any],
    *,
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    dipper_cfg: DipperConfig,
    metric_cfg: DatasetConfig,
    dry_run: bool,
) -> dict[str, Any] | None:
    task_id = str(row.get("id", ""))
    try:
        mixed_text = str(row.get("mixed_text", ""))
        original_text = str(row.get("original_text", ""))
        labels = _normalize_labels(row.get("sentence_labels", []), task_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[%s] 输入字段非法，已跳过: %s", task_id, exc)
        return None

    if not mixed_text.strip():
        logger.warning("[%s] mixed_text 为空，已跳过。", task_id)
        return None

    sentences = split_into_sentences(mixed_text)
    if len(sentences) != len(labels):
        logger.warning(
            "[%s] mixed_text 分句数与 sentence_labels 不一致 (sentences=%d, labels=%d)，已跳过。",
            task_id,
            len(sentences),
            len(labels),
        )
        return None

    if int(row.get("n_sentences", len(labels))) != len(labels):
        logger.warning(
            "[%s] n_sentences 与 sentence_labels 长度不一致 (n_sentences=%s, labels=%d)，已跳过。",
            task_id,
            row.get("n_sentences"),
            len(labels),
        )
        return None

    selected_indices = [index for index, label in enumerate(labels) if label == 1]
    final_sentences = list(sentences)
    ai_sentences_original = [sentences[index] for index in selected_indices]
    ai_sentences_rewritten: list[str] = []

    if selected_indices and not dry_run:
        for index in selected_indices:
            try:
                rewritten = await _call_dipper(
                    client,
                    sem,
                    text=sentences[index],
                    cfg=dipper_cfg,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("[%s] DIPPER 调用失败，已跳过: %s", task_id, exc)
                return None
            if len(split_into_sentences(rewritten)) != 1:
                logger.warning(
                    "[%s] DIPPER 改写后被切分为 %d 句，已跳过。",
                    task_id,
                    len(split_into_sentences(rewritten)),
                )
                return None
            final_sentences[index] = rewritten
            ai_sentences_rewritten.append(rewritten)
    else:
        ai_sentences_rewritten = [sentences[index] for index in selected_indices]

    rewritten_text = " ".join(final_sentences)
    final_sentence_count = len(split_into_sentences(rewritten_text))
    if final_sentence_count != len(sentences):
        logger.warning(
            "[%s] 回填后句子数发生变化（before=%d, after=%d），已跳过。",
            task_id,
            len(sentences),
            final_sentence_count,
        )
        return None

    labels_dict = compute_labels(
        original_text,
        final_sentences,
        [i for i, label in enumerate(labels) if label == 1],
        metric_cfg,
    )
    output = dict(row)
    output["original_mixed_text"] = mixed_text
    output["mixed_text"] = rewritten_text
    output["rewritten_text"] = rewritten_text
    output["ai_sentences_original"] = ai_sentences_original
    output["ai_sentences_rewritten"] = ai_sentences_rewritten
    output["rewrite_info"] = {
        **_build_prompt_meta(dipper_cfg.base_url, dipper_cfg.lex, dipper_cfg.order, dipper_cfg.max_new_tokens),
        "timestamp": datetime.now().date().isoformat(),
        "status": "dry_run" if dry_run else "ok",
    }
    output["lir"] = labels_dict["lir"]
    output["jaccard_distance"] = labels_dict["jaccard_distance"]
    output["sentence_jaccard"] = labels_dict["sentence_jaccard"]
    output["cosine_distance"] = labels_dict["cosine_distance"]
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rewrite benchmark_grouped/test.jsonl with a DIPPER HTTP service.",
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
        "--base-url",
        type=str,
        default="http://127.0.0.1:6006",
        help="DIPPER service base URL.",
    )
    parser.add_argument(
        "--lex",
        type=int,
        default=40,
        help="DIPPER lex similarity code.",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=100,
        help="DIPPER order similarity code.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=80,
        help="Maximum output tokens for one paraphrase call.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--concurrent-requests",
        type=int,
        default=4,
        help="Maximum concurrent HTTP requests.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Only process the first N records.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip DIPPER calls and keep selected sentences unchanged.",
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

    rows = _load_rows(input_path)
    if args.max_records is not None:
        rows = rows[: max(0, args.max_records)]

    dipper_cfg = DipperConfig(
        base_url=args.base_url,
        lex=args.lex,
        order=args.order,
        max_new_tokens=args.max_new_tokens,
        timeout_seconds=args.timeout,
        concurrent_requests=max(1, args.concurrent_requests),
    )

    logger.info("DIPPER rewrite config:")
    logger.info("  input_path        = %s", input_path)
    logger.info("  output_path       = %s", output_path)
    logger.info("  base_url          = %s", dipper_cfg.base_url)
    logger.info("  lex               = %s", dipper_cfg.lex)
    logger.info("  order             = %s", dipper_cfg.order)
    logger.info("  max_new_tokens    = %s", dipper_cfg.max_new_tokens)
    logger.info("  concurrent        = %s", dipper_cfg.concurrent_requests)
    logger.info("  max_records       = %s", args.max_records)
    logger.info("  dry_run           = %s", args.dry_run)

    sem = asyncio.Semaphore(dipper_cfg.concurrent_requests)
    timeout = httpx.Timeout(dipper_cfg.timeout_seconds)
    limits = httpx.Limits(max_connections=dipper_cfg.concurrent_requests, max_keepalive_connections=dipper_cfg.concurrent_requests)
    metric_cfg = DatasetConfig(
        rewrite_model="dipper-http",
        source_path=str(input_path),
        output_dir=str(output_path.parent),
    )

    async with httpx.AsyncClient(
        base_url=dipper_cfg.base_url.rstrip("/"),
        timeout=timeout,
        limits=limits,
    ) as client:
        if not args.dry_run:
            health = await client.get("/health")
            health.raise_for_status()

        tasks = [
            _rewrite_row(
                row,
                client=client,
                sem=sem,
                dipper_cfg=dipper_cfg,
                metric_cfg=metric_cfg,
                dry_run=args.dry_run,
            )
            for row in rows
        ]
        results = await asyncio.gather(*tasks)

    written = 0
    skipped = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for result in results:
            if result is None:
                skipped += 1
                continue
            fh.write(json.dumps(result, ensure_ascii=False) + "\n")
            written += 1

    logger.info("Done: written=%d skipped=%d output=%s", written, skipped, output_path)


if __name__ == "__main__":
    asyncio.run(main())
