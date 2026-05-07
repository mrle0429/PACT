#!/usr/bin/env python3
"""
Apply the official mixed grouped benchmark split to rewrite datasets.

This script does not create a new random split. It treats the existing
datasplit/benchmark_grouped/{train,val,test}.jsonl files as the source of truth
for human_id -> split assignment, then assigns rewrite rows by the same
human_id. This keeps all variants derived from one original human document in
the same split across mixed and rewrite datasets.

Default policy:
- use only the same 8 in-distribution benchmark models as the mixed split;
- keep every rewrite row, including 0% rows with status=no_selected_sentences;
- normalize the one known source-model typo field, sourwrce_model -> source_model;
- write train.jsonl, val.jsonl, test.jsonl, and summary.json under
  datasplit/rewrite_grouped/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.label_calculator import (
    compute_cosine_distance,
    compute_jaccard_distance,
    compute_lir,
    compute_sentence_jaccard,
)
from src.sentence_processor import split_into_sentences


DEFAULT_REWRITE_DATASETS = [
    "output/rewrite_qwen3.5-flash.jsonl",
    "output/rewrite_qwen3.5-plus.jsonl",
    "output/rewrite_qwen3.6-plus.jsonl",
    "output/rewrite_DeepSeek-V3.2.jsonl",
    "output/rewrite_MiniMax-M2.7.jsonl",
    "output/rewrite_doubao-seed-2-0-pro.jsonl",
    "output/rewrite_gemini-3.1-flash-lite-preview.jsonl",
    "output/rewrite_llama4-fast:latest.jsonl",
]

MIXED_SPLIT_DIR = "datasplit/benchmark_grouped"
DEFAULT_OUTPUT_DIR = "datasplit/rewrite_grouped"
RATIO_SUFFIX_RE = re.compile(r"^(?P<human_id>.+)_r(?:0|20|40|60|80|100)_(?P<mode>.+)$")
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assign rewrite JSONL rows using the official mixed grouped split."
    )
    parser.add_argument(
        "--rewrite-datasets",
        nargs="+",
        default=DEFAULT_REWRITE_DATASETS,
        help="Rewrite JSONL files to split. Defaults to the 8 benchmark rewrite models.",
    )
    parser.add_argument(
        "--mixed-split-dir",
        default=MIXED_SPLIT_DIR,
        help="Directory containing the official mixed train/val/test JSONL files.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for rewrite train.jsonl, val.jsonl, test.jsonl, and summary.json.",
    )
    parser.add_argument(
        "--status",
        choices=["all", "ok"],
        default="all",
        help="Keep all rewrite rows, or only rows with rewrite_info.status == ok.",
    )
    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Do not add mixed_* and rewrite_* metric fields to output rows.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing rewrite split output directory.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def original_text_hash(row: dict[str, Any]) -> str:
    text = row.get("original_text") or row.get("text")
    if not isinstance(text, str) or not text:
        raise ValueError(f"Cannot infer human_id for row without id/original_text/text: {row}")
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    return f"text_sha1_{digest}"


def infer_human_id(row: dict[str, Any]) -> str:
    row_id = row.get("id")
    if isinstance(row_id, str):
        match = RATIO_SUFFIX_RE.match(row_id)
        if match:
            return match.group("human_id")
        return row_id
    return original_text_hash(row)


def ratio_key(value: Any) -> str:
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return str(value)


def load_split_map(split_dir: Path) -> dict[str, str]:
    split_by_human_id: dict[str, str] = {}

    for split in SPLITS:
        path = split_dir / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing mixed split file: {path}")

        for row in read_jsonl(path):
            human_id = row.get("human_id") or infer_human_id(row)
            if not isinstance(human_id, str) or not human_id:
                raise ValueError(f"Invalid human_id in {path}: {row}")
            previous = split_by_human_id.get(human_id)
            if previous is not None and previous != split:
                raise RuntimeError(
                    f"human_id {human_id!r} appears in both {previous!r} and {split!r}"
                )
            split_by_human_id[human_id] = split

    return split_by_human_id


def source_model(row: dict[str, Any]) -> str:
    value = row.get("source_model") or row.get("rewrite_model")
    if value is not None:
        return str(value)
    rewrite_info = row.get("rewrite_info")
    if isinstance(rewrite_info, dict) and rewrite_info.get("rewriter") is not None:
        return str(rewrite_info["rewriter"])
    return "unknown"


def load_mixed_metrics(split_dir: Path) -> dict[tuple[str, str], dict[str, float | None]]:
    metrics_by_key: dict[tuple[str, str], dict[str, float | None]] = {}

    for split in SPLITS:
        path = split_dir / f"{split}.jsonl"
        for row in read_jsonl(path):
            row_id = row.get("id")
            if not isinstance(row_id, str):
                continue
            metrics_by_key[(row_id, source_model(row))] = {
                "mixed_lir": row.get("lir"),
                "mixed_jaccard_distance": row.get("jaccard_distance"),
                "mixed_sentence_jaccard": row.get("sentence_jaccard"),
                "mixed_cosine_distance": row.get("cosine_distance"),
            }

    return metrics_by_key


def ai_indices_from_labels(sentence_labels: Any) -> list[int]:
    if not isinstance(sentence_labels, list):
        return []
    return [idx for idx, label in enumerate(sentence_labels) if int(label) == 1]


def compute_text_metrics(
    original_text: str,
    variant_text: str,
    sentence_labels: list[int],
) -> tuple[dict[str, float | None], bool]:
    variant_sentences = split_into_sentences(variant_text)
    aligned = len(variant_sentences) == len(sentence_labels)
    ai_indices = ai_indices_from_labels(sentence_labels)

    lir = compute_lir(ai_indices, variant_sentences) if aligned else None
    return (
        {
            "lir": lir,
            "jaccard_distance": compute_jaccard_distance(original_text, variant_text),
            "sentence_jaccard": compute_sentence_jaccard(
                original_text,
                variant_text,
                sentence_labels,
            ),
            "cosine_distance": compute_cosine_distance(original_text, variant_text),
        },
        aligned,
    )


def add_metric_fields(
    row: dict[str, Any],
    mixed_metrics_by_key: dict[tuple[str, str], dict[str, float | None]],
) -> tuple[dict[str, Any], dict[str, int]]:
    output = dict(row)
    stats = {
        "mixed_metrics_from_source": 0,
        "mixed_metrics_recomputed": 0,
        "mixed_lir_alignment_failures": 0,
        "rewrite_lir_alignment_failures": 0,
    }

    row_id = str(output.get("id", ""))
    model = source_model(output)
    mixed_metrics = mixed_metrics_by_key.get((row_id, model))

    # 0% rows are deduplicated as "human" in the mixed benchmark, but rewrite
    # files still carry their source model name. Recomputing gives the exact
    # zero-valued mixed metrics for those rows without relying on model matching.
    if mixed_metrics is None:
        mixed_raw, mixed_aligned = compute_text_metrics(
            original_text=str(output.get("original_text", "")),
            variant_text=str(output.get("mixed_text", "")),
            sentence_labels=output.get("sentence_labels", []),
        )
        mixed_metrics = {
            "mixed_lir": mixed_raw["lir"],
            "mixed_jaccard_distance": mixed_raw["jaccard_distance"],
            "mixed_sentence_jaccard": mixed_raw["sentence_jaccard"],
            "mixed_cosine_distance": mixed_raw["cosine_distance"],
        }
        stats["mixed_metrics_recomputed"] += 1
        stats["mixed_lir_alignment_failures"] += int(not mixed_aligned)
    else:
        stats["mixed_metrics_from_source"] += 1

    output.update(mixed_metrics)

    rewrite_raw, rewrite_aligned = compute_text_metrics(
        original_text=str(output.get("original_text", "")),
        variant_text=str(output.get("rewritten_text", "")),
        sentence_labels=output.get("sentence_labels", []),
    )
    output.update(
        {
            "rewrite_lir": rewrite_raw["lir"],
            "rewrite_jaccard_distance": rewrite_raw["jaccard_distance"],
            "rewrite_sentence_jaccard": rewrite_raw["sentence_jaccard"],
            "rewrite_cosine_distance": rewrite_raw["cosine_distance"],
        }
    )
    stats["rewrite_lir_alignment_failures"] += int(not rewrite_aligned)

    return output, stats


def rewrite_status(row: dict[str, Any]) -> str:
    rewrite_info = row.get("rewrite_info")
    if isinstance(rewrite_info, dict):
        return str(rewrite_info.get("status", "missing"))
    return "missing"


def normalize_source_model(row: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    normalized = dict(row)
    corrected_typo = False

    if "source_model" not in normalized and "sourwrce_model" in normalized:
        normalized["source_model"] = normalized["sourwrce_model"]
        del normalized["sourwrce_model"]
        corrected_typo = True

    if "source_model" not in normalized:
        rewrite_info = normalized.get("rewrite_info")
        if isinstance(rewrite_info, dict) and "rewriter" in rewrite_info:
            normalized["source_model"] = rewrite_info["rewriter"]

    return normalized, corrected_typo


def validate_no_group_leakage(rows_by_split: dict[str, list[dict[str, Any]]]) -> None:
    locations: dict[str, set[str]] = defaultdict(set)
    for split, rows in rows_by_split.items():
        for row in rows:
            locations[row["human_id"]].add(split)

    leaked = {human_id: splits for human_id, splits in locations.items() if len(splits) > 1}
    if leaked:
        examples = list(leaked.items())[:10]
        raise RuntimeError(f"Found human_id leakage across rewrite splits: {examples}")


def summarize(
    rows_by_split: dict[str, list[dict[str, Any]]],
    split_by_human_id: dict[str, str],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "source_split_dir": metadata["source_split_dir"],
        "output_dir": metadata["output_dir"],
        "status_filter": metadata["status_filter"],
        "input_files": metadata["input_files"],
        "source_split_human_id_counts": dict(Counter(split_by_human_id.values())),
        "row_counts": {split: len(rows_by_split[split]) for split in SPLITS},
        "human_id_counts": {
            split: len({row["human_id"] for row in rows_by_split[split]}) for split in SPLITS
        },
        "by_target_ai_ratio": {},
        "by_rewrite_status": {},
        "by_source_model": {},
        "by_source_file": {},
        "skipped_by_status": metadata["skipped_by_status"],
        "skipped_missing_split": metadata["skipped_missing_split"],
        "corrected_sourwrce_model_rows": metadata["corrected_sourwrce_model_rows"],
        "metrics_added": metadata["metrics_added"],
        "metric_source_counts": metadata["metric_source_counts"],
    }

    for split in SPLITS:
        rows = rows_by_split[split]
        summary["by_target_ai_ratio"][split] = dict(
            sorted(Counter(ratio_key(row.get("target_ai_ratio")) for row in rows).items())
        )
        summary["by_rewrite_status"][split] = dict(
            sorted(Counter(rewrite_status(row) for row in rows).items())
        )
        summary["by_source_model"][split] = dict(
            sorted(Counter(str(row.get("source_model", "unknown")) for row in rows).items())
        )
        summary["by_source_file"][split] = dict(
            sorted(Counter(str(row.get("_source_file", "unknown")) for row in rows).items())
        )

    return summary


def main() -> None:
    args = parse_args()
    split_dir = Path(args.mixed_split_dir)
    output_dir = Path(args.output_dir)

    existing_outputs = [output_dir / f"{split}.jsonl" for split in SPLITS]
    existing_outputs.append(output_dir / "summary.json")
    if not args.overwrite and any(path.exists() for path in existing_outputs):
        raise FileExistsError(
            f"Output rewrite split files already exist in {output_dir}. "
            "Use --overwrite to replace them."
        )

    split_by_human_id = load_split_map(split_dir)
    mixed_metrics_by_key = load_mixed_metrics(split_dir)
    rows_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    input_counts: dict[str, int] = {}
    skipped_by_status = 0
    skipped_missing_split = 0
    corrected_typo_rows = 0
    metric_source_counts: Counter[str] = Counter()

    for item in args.rewrite_datasets:
        path = Path(item)
        if not path.exists():
            raise FileNotFoundError(f"Input rewrite dataset not found: {path}")

        rows = read_jsonl(path)
        input_counts[str(path)] = len(rows)
        for row in rows:
            if args.status == "ok" and rewrite_status(row) != "ok":
                skipped_by_status += 1
                continue

            normalized, corrected_typo = normalize_source_model(row)
            corrected_typo_rows += int(corrected_typo)

            human_id = infer_human_id(normalized)
            split = split_by_human_id.get(human_id)
            if split is None:
                skipped_missing_split += 1
                continue

            output_row = dict(normalized)
            output_row["human_id"] = human_id
            output_row["split"] = split
            output_row["_source_file"] = path.name
            if not args.no_metrics:
                output_row, metric_stats = add_metric_fields(output_row, mixed_metrics_by_key)
                metric_source_counts.update(metric_stats)
            rows_by_split[split].append(output_row)

    validate_no_group_leakage(rows_by_split)

    output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        write_jsonl(output_dir / f"{split}.jsonl", rows_by_split[split])

    summary = summarize(
        rows_by_split=rows_by_split,
        split_by_human_id=split_by_human_id,
        metadata={
            "source_split_dir": str(split_dir),
            "output_dir": str(output_dir),
            "status_filter": args.status,
            "input_files": input_counts,
            "skipped_by_status": skipped_by_status,
            "skipped_missing_split": skipped_missing_split,
            "corrected_sourwrce_model_rows": corrected_typo_rows,
            "metrics_added": not args.no_metrics,
            "metric_source_counts": dict(sorted(metric_source_counts.items())),
        },
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote rewrite grouped split to {output_dir}")
    print(json.dumps(summary["human_id_counts"], ensure_ascii=False, sort_keys=True))
    print(json.dumps(summary["row_counts"], ensure_ascii=False, sort_keys=True))
    print(f"Skipped rows without mixed split group: {skipped_missing_split}")
    print(f"Skipped rows by status filter: {skipped_by_status}")
    print(f"Corrected sourwrce_model rows: {corrected_typo_rows}")
    if not args.no_metrics:
        print(json.dumps(summary["metric_source_counts"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
