#!/usr/bin/env python3
"""
Create a human-ID grouped benchmark split for PACT mixed datasets.

The split unit is the original human document, not an individual mixed row.
All variants derived from the same human text are assigned to the same split.
For 0% AI rows, only one human row is kept per human_id because each model
dataset usually contains the same human-only sample.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_DATASETS = [
    "datasets/mixed_dataset_qwen3.5-flash.jsonl",
    "datasets/mixed_dataset_qwen3.5-plus.jsonl",
    "datasets/mixed_dataset_qwen3.6-plus.jsonl",
    "datasets/mixed_dataset_DeepSeek-V3.2.jsonl",
    "datasets/mixed_dataset_MiniMax-M2.7.jsonl",
    "datasets/mixed_dataset_doubao-seed-2-0-pro.jsonl",
    "datasets/mixed_dataset_gemini-3.1-flash-lite-preview.jsonl",
    "datasets/mixed_dataset_llama4-fast:latest.jsonl",
]

RATIO_SUFFIX_RE = re.compile(r"^(?P<human_id>.+)_r(?:0|20|40|60|80|100)_(?P<mode>.+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a leakage-safe grouped benchmark split by original human text."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Input mixed JSONL files. Defaults to the 8 benchmark models.",
    )
    parser.add_argument(
        "--human-file",
        default="datasets/human_texts_1k.cleaned.jsonl",
        help=(
            "Optional human seed JSONL used to define split groups and stable order. "
            "Set to an empty string to split by human_ids found in mixed files only."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="datasplit/benchmark_grouped",
        help="Output directory for train.jsonl, val.jsonl, test.jsonl, and summary.json.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument(
        "--no-coverage-stratify",
        action="store_true",
        help=(
            "Disable stratification by available source-model coverage. By default, "
            "human_ids with the same available model set are split together so "
            "partially generated model files still appear in train/val/test."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing output directory with split files.",
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


def load_human_ids(human_file: str) -> list[str]:
    if not human_file:
        return []

    path = Path(human_file)
    if not path.exists():
        print(f"[warn] human file not found, falling back to mixed-file groups: {path}")
        return []

    human_ids: list[str] = []
    seen: set[str] = set()
    for row in read_jsonl(path):
        human_id = infer_human_id(row)
        if human_id not in seen:
            seen.add(human_id)
            human_ids.append(human_id)
    return human_ids


def load_mixed_rows(paths: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    input_counts: dict[str, int] = {}

    for item in paths:
        path = Path(item)
        if not path.exists():
            raise FileNotFoundError(f"Input dataset not found: {path}")

        rows = read_jsonl(path)
        input_counts[str(path)] = len(rows)
        for row in rows:
            human_id = infer_human_id(row)
            enriched = dict(row)
            enriched["human_id"] = human_id
            enriched["source_model"] = enriched.get("rewrite_model", "unknown")
            enriched["_source_file"] = path.name
            all_rows.append(enriched)

    return all_rows, {"input_files": input_counts}


def dedupe_human_only_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    kept: list[dict[str, Any]] = []
    seen_human_only: set[str] = set()
    dropped = 0

    for row in rows:
        is_human_only = ratio_key(row.get("target_ai_ratio")) == "0.0"
        if is_human_only:
            human_id = row["human_id"]
            if human_id in seen_human_only:
                dropped += 1
                continue
            seen_human_only.add(human_id)
        kept.append(row)

    return kept, dropped


def make_group_split(
    human_ids: list[str],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    strata_by_human_id: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, str]:
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

    split_by_human_id: dict[str, str] = {}
    rng = random.Random(seed)

    if strata_by_human_id:
        buckets: dict[tuple[str, ...], list[str]] = defaultdict(list)
        for human_id in human_ids:
            buckets[strata_by_human_id.get(human_id, ("__missing__",))].append(human_id)
        split_units = list(buckets.values())
    else:
        split_units = [list(human_ids)]

    n_all = len(human_ids)
    target_counts = {
        "train": int(n_all * train_ratio),
        "val": int(n_all * val_ratio),
    }
    target_counts["test"] = n_all - target_counts["train"] - target_counts["val"]

    allocations: list[dict[str, int]] = []
    remainders: list[tuple[float, int, str]] = []
    for bucket_index, bucket in enumerate(split_units):
        n_bucket = len(bucket)
        train_ideal = n_bucket * train_ratio
        val_ideal = n_bucket * val_ratio
        test_ideal = n_bucket * test_ratio
        allocation = {
            "train": int(train_ideal),
            "val": int(val_ideal),
            "test": int(test_ideal),
        }
        allocations.append(allocation)
        remainders.extend(
            [
                (train_ideal - allocation["train"], bucket_index, "train"),
                (val_ideal - allocation["val"], bucket_index, "val"),
                (test_ideal - allocation["test"], bucket_index, "test"),
            ]
        )

    assigned_counts = {
        split: sum(allocation[split] for allocation in allocations)
        for split in ("train", "val", "test")
    }
    remaining_counts = {
        split: target_counts[split] - assigned_counts[split]
        for split in ("train", "val", "test")
    }

    for _fraction, bucket_index, split in sorted(remainders, reverse=True):
        if remaining_counts[split] <= 0:
            continue
        if sum(allocations[bucket_index].values()) >= len(split_units[bucket_index]):
            continue
        allocations[bucket_index][split] += 1
        remaining_counts[split] -= 1

    if any(count != 0 for count in remaining_counts.values()):
        raise RuntimeError(f"Could not allocate split counts exactly: {remaining_counts}")

    for bucket, allocation in zip(split_units, allocations):
        shuffled = list(bucket)
        rng.shuffle(shuffled)

        n_train = allocation["train"]
        n_val = allocation["val"]

        for human_id in shuffled[:n_train]:
            split_by_human_id[human_id] = "train"
        for human_id in shuffled[n_train : n_train + n_val]:
            split_by_human_id[human_id] = "val"
        for human_id in shuffled[n_train + n_val :]:
            split_by_human_id[human_id] = "test"

    return split_by_human_id


def summarize(rows_by_split: dict[str, list[dict[str, Any]]], split_by_human_id: dict[str, str]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "human_id_counts": dict(Counter(split_by_human_id.values())),
        "row_counts": {split: len(rows) for split, rows in rows_by_split.items()},
        "by_target_ai_ratio": {},
        "by_source_model": {},
        "by_source_file": {},
    }

    for split, rows in rows_by_split.items():
        summary["by_target_ai_ratio"][split] = dict(
            sorted(Counter(ratio_key(row.get("target_ai_ratio")) for row in rows).items())
        )
        summary["by_source_model"][split] = dict(
            sorted(Counter(str(row.get("source_model", "unknown")) for row in rows).items())
        )
        summary["by_source_file"][split] = dict(
            sorted(Counter(str(row.get("_source_file", "unknown")) for row in rows).items())
        )

    return summary


def validate_no_group_leakage(rows_by_split: dict[str, list[dict[str, Any]]]) -> None:
    locations: dict[str, set[str]] = defaultdict(set)
    for split, rows in rows_by_split.items():
        for row in rows:
            locations[row["human_id"]].add(split)

    leaked = {human_id: splits for human_id, splits in locations.items() if len(splits) > 1}
    if leaked:
        examples = list(leaked.items())[:10]
        raise RuntimeError(f"Found human_id leakage across splits: {examples}")


def build_coverage_strata(rows: list[dict[str, Any]]) -> dict[str, tuple[str, ...]]:
    coverage: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if ratio_key(row.get("target_ai_ratio")) == "0.0":
            continue
        coverage[row["human_id"]].add(str(row.get("source_model", "unknown")))
    return {human_id: tuple(sorted(models)) for human_id, models in coverage.items()}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    existing_outputs = [output_dir / name for name in ("train.jsonl", "val.jsonl", "test.jsonl")]
    if not args.overwrite and any(path.exists() for path in existing_outputs):
        raise FileExistsError(
            f"Output split files already exist in {output_dir}. Use --overwrite to replace them."
        )

    rows, metadata = load_mixed_rows(args.datasets)
    rows, dropped_human_duplicates = dedupe_human_only_rows(rows)

    human_ids = load_human_ids(args.human_file)
    mixed_human_ids = sorted({row["human_id"] for row in rows})
    if human_ids:
        mixed_human_id_set = set(mixed_human_ids)
        human_ids = [human_id for human_id in human_ids if human_id in mixed_human_id_set]
    else:
        human_ids = mixed_human_ids

    coverage_strata = build_coverage_strata(rows)
    split_by_human_id = make_group_split(
        human_ids=human_ids,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        strata_by_human_id=None if args.no_coverage_stratify else coverage_strata,
    )

    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    skipped_rows = 0
    for row in rows:
        split = split_by_human_id.get(row["human_id"])
        if split is None:
            skipped_rows += 1
            continue
        output_row = dict(row)
        output_row["split"] = split
        rows_by_split[split].append(output_row)

    validate_no_group_leakage(rows_by_split)

    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        write_jsonl(output_dir / f"{split}.jsonl", rows_by_split[split])

    summary = summarize(rows_by_split, split_by_human_id)
    summary["seed"] = args.seed
    summary["ratios"] = {
        "train": args.train_ratio,
        "val": args.val_ratio,
        "test": args.test_ratio,
    }
    summary["dropped_duplicate_human_only_rows"] = dropped_human_duplicates
    summary["skipped_rows_without_split_group"] = skipped_rows
    summary["coverage_stratified"] = not args.no_coverage_stratify
    summary["coverage_strata_counts"] = {
        "|".join(key): value
        for key, value in sorted(Counter(coverage_strata.get(human_id, ("__missing__",)) for human_id in human_ids).items())
    }
    summary.update(metadata)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote grouped benchmark split to {output_dir}")
    print(json.dumps(summary["human_id_counts"], ensure_ascii=False, sort_keys=True))
    print(json.dumps(summary["row_counts"], ensure_ascii=False, sort_keys=True))
    print(f"Dropped duplicate 0% human rows: {dropped_human_duplicates}")


if __name__ == "__main__":
    main()
