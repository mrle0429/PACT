#!/usr/bin/env python3
"""
Analyze mixed-dataset JSONL files and export tables, charts, and a Markdown report.

Examples:
    python scripts/analyze_dataset_statistics.py
    python scripts/analyze_dataset_statistics.py --inputs output/mixed_dataset_claude-haiku-4.5.jsonl
    python scripts/analyze_dataset_statistics.py --inputs output/mixed_dataset_*.jsonl --source-human data/human_texts_1k.cleaned.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = PROJECT_ROOT / ".cache"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.sentence_processor import split_into_sentences
except Exception:
    _FALLBACK_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

    def split_into_sentences(text: str) -> list[str]:
        return [sentence.strip() for sentence in _FALLBACK_SENTENCE_RE.split(text.strip()) if sentence.strip()]

DEFAULT_INPUTS = ["output/mixed_dataset_*.jsonl"]
DEFAULT_SOURCE_HUMAN = PROJECT_ROOT / "data" / "human_texts_1k.cleaned.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "analysis" / "dataset_statistics"

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")
RECORD_ID_RE = re.compile(r"^(?P<doc_id>.+)_r\d+_(?:block|scatter)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze mixed dataset statistics")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=DEFAULT_INPUTS,
        help="Input JSONL paths or glob patterns",
    )
    parser.add_argument(
        "--source-human",
        type=Path,
        default=DEFAULT_SOURCE_HUMAN,
        help="Optional source human JSONL used to compute coverage and stable splits",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save tables, plots, and report",
    )
    parser.add_argument(
        "--figure-dpi",
        type=int,
        default=220,
        help="DPI for saved figures",
    )
    return parser.parse_args()


def expand_input_patterns(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(PROJECT_ROOT.glob(pattern))
        if matches:
            paths.extend(matches)
            continue
        candidate = Path(pattern)
        if candidate.exists():
            paths.append(candidate.resolve())

    unique_paths = sorted({path.resolve() for path in paths})
    if not unique_paths:
        raise FileNotFoundError(f"No input files matched: {patterns}")
    return unique_paths


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def infer_run_name(path: Path) -> str:
    prefix = "mixed_dataset_"
    stem = path.stem
    return stem[len(prefix):] if stem.startswith(prefix) else stem


def infer_source_doc_id(record_id: str) -> str:
    match = RECORD_ID_RE.match(record_id)
    return match.group("doc_id") if match else record_id


def infer_llm_family(model_name: str) -> str:
    normalized = model_name.lower()
    if normalized == "human":
        return "human"
    if "claude" in normalized:
        return "claude"
    if "deepseek" in normalized:
        return "deepseek"
    if "doubao" in normalized:
        return "doubao"
    if "gemini" in normalized:
        return "gemini"
    if "gemma" in normalized:
        return "gemma"
    if "llama" in normalized:
        return "llama"
    if "minimax" in normalized:
        return "minimax"
    if "qwen" in normalized:
        return "qwen"
    return model_name.split("-", 1)[0]


def tokenize_words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def type_token_ratio(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def normalized_word_edit_distance(tokens_a: list[str], tokens_b: list[str]) -> float:
    if tokens_a == tokens_b:
        return 0.0
    if not tokens_a and not tokens_b:
        return 0.0
    if not tokens_a or not tokens_b:
        return 1.0

    if len(tokens_a) < len(tokens_b):
        tokens_a, tokens_b = tokens_b, tokens_a

    previous = list(range(len(tokens_b) + 1))
    for i, token_a in enumerate(tokens_a, 1):
        current = [i]
        for j, token_b in enumerate(tokens_b, 1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (token_a != token_b)
            current.append(min(insertion, deletion, substitution))
        previous = current

    distance = previous[-1]
    return distance / max(len(tokens_a), len(tokens_b))


def compute_contiguous_ai_spans(labels: list[int]) -> list[int]:
    spans: list[int] = []
    current = 0
    for label in labels:
        if label == 1:
            current += 1
        elif current:
            spans.append(current)
            current = 0
    if current:
        spans.append(current)
    return spans


def word_bucket(word_count: int) -> str:
    bins = [
        (0, 119, "<120"),
        (120, 159, "120-159"),
        (160, 199, "160-199"),
        (200, 239, "200-239"),
        (240, 299, "240-299"),
        (300, math.inf, "300+"),
    ]
    for low, high, label in bins:
        if low <= word_count <= high:
            return label
    return "unknown"


def lir_bucket(value: float) -> str:
    bins = [
        (0.0, 0.05, "0.00-0.05"),
        (0.05, 0.25, "0.05-0.25"),
        (0.25, 0.45, "0.25-0.45"),
        (0.45, 0.65, "0.45-0.65"),
        (0.65, 0.85, "0.65-0.85"),
        (0.85, 1.01, "0.85-1.00"),
    ]
    for low, high, label in bins:
        if low <= value < high or (label == "0.85-1.00" and value <= 1.0):
            return label
    return "unknown"


def ratio_label(value: float) -> str:
    return f"{int(round(value * 100)):d}%"


def safe_mean(series: pd.Series) -> float:
    cleaned = series.dropna()
    if cleaned.empty:
        return 0.0
    return float(cleaned.mean())


def build_stable_split_map(doc_ids: list[str], train_ratio: float = 0.8, val_ratio: float = 0.1) -> dict[str, str]:
    hashed = sorted(
        (
            hashlib.md5(doc_id.encode("utf-8")).hexdigest(),
            doc_id,
        )
        for doc_id in set(doc_ids)
    )
    total = len(hashed)
    train_cut = int(round(total * train_ratio))
    val_cut = train_cut + int(round(total * val_ratio))

    split_map: dict[str, str] = {}
    for idx, (_, doc_id) in enumerate(hashed):
        if idx < train_cut:
            split_map[doc_id] = "train"
        elif idx < val_cut:
            split_map[doc_id] = "validation"
        else:
            split_map[doc_id] = "test"
    return split_map


def load_source_human_metadata(path: Path | None) -> tuple[pd.DataFrame, dict[str, str]]:
    if path is None or not path.exists():
        return pd.DataFrame(), {}

    rows: list[dict] = []
    for record in read_jsonl(path):
        doc_id = str(record.get("id", ""))
        if not doc_id:
            continue
        prefix = doc_id.split("_", 1)[0]
        if prefix == "arxiv":
            source_domain = "academic"
            source_dataset = "arxiv"
        elif prefix == "owt":
            source_domain = "web"
            source_dataset = "openwebtext"
        elif prefix == "xsum":
            source_domain = "news"
            source_dataset = "xsum"
        elif prefix == "daigt":
            source_domain = "essay"
            source_dataset = "daigt"
        else:
            source_domain = "unknown"
            source_dataset = "unknown"

        rows.append(
            {
                "source_doc_id": doc_id,
                "source_dataset": source_dataset,
                "source_domain": source_domain,
                "source_sentence_count": int(record.get("sentence_count", 0) or 0),
                "source_word_count": len(tokenize_words(str(record.get("text", "")))),
            }
        )

    frame = pd.DataFrame(rows)
    split_map = build_stable_split_map(frame["source_doc_id"].tolist()) if not frame.empty else {}
    return frame, split_map


def build_record_rows(input_paths: list[Path], split_map: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_rows: list[dict] = []
    dedup_rows: list[dict] = []
    seen_keys: set[tuple[str, str]] = set()

    for path in input_paths:
        run_name = infer_run_name(path)
        for record in read_jsonl(path):
            record_id = str(record["id"])
            rewrite_model = str(record.get("rewrite_model", "unknown"))
            source_doc_id = infer_source_doc_id(record_id)
            split_name = split_map.get(source_doc_id, "unknown")
            labels = list(record.get("sentence_labels", []))
            original_text = str(record.get("original_text", ""))
            mixed_text = str(record.get("mixed_text", ""))

            original_tokens = tokenize_words(original_text)
            mixed_tokens = tokenize_words(mixed_text)
            original_sentences = split_into_sentences(original_text)
            mixed_sentences = split_into_sentences(mixed_text)

            aligned = len(original_sentences) == len(mixed_sentences) == len(labels)
            ai_sentence_indices = [idx for idx, label in enumerate(labels) if label == 1]
            exact_copy_ai_sentences = 0
            rewrite_distances: list[float] = []

            if aligned and ai_sentence_indices:
                for idx in ai_sentence_indices:
                    original_sentence = original_sentences[idx].strip()
                    mixed_sentence = mixed_sentences[idx].strip()
                    if original_sentence == mixed_sentence:
                        exact_copy_ai_sentences += 1
                    rewrite_distances.append(
                        normalized_word_edit_distance(
                            tokenize_words(original_sentence),
                            tokenize_words(mixed_sentence),
                        )
                    )

            spans = compute_contiguous_ai_spans(labels)

            row = {
                "input_file": path.name,
                "run_name": run_name,
                "record_id": record_id,
                "source_doc_id": source_doc_id,
                "source_dataset": str(record.get("source_dataset", "unknown")),
                "source_domain": str(record.get("source_domain", "unknown")),
                "split": split_name,
                "rewrite_model": rewrite_model,
                "llm_family": infer_llm_family(rewrite_model),
                "target_ai_ratio": float(record.get("target_ai_ratio", 0.0) or 0.0),
                "target_ai_ratio_label": ratio_label(float(record.get("target_ai_ratio", 0.0) or 0.0)),
                "mixing_mode": str(record.get("mixing_mode", "unknown")),
                "n_sentences": int(record.get("n_sentences", len(labels)) or len(labels)),
                "original_word_count": len(original_tokens),
                "mixed_word_count": len(mixed_tokens),
                "avg_original_sentence_words": (len(original_tokens) / len(original_sentences)) if original_sentences else 0.0,
                "avg_mixed_sentence_words": (len(mixed_tokens) / len(mixed_sentences)) if mixed_sentences else 0.0,
                "length_change_ratio": (len(mixed_tokens) / len(original_tokens)) if original_tokens else None,
                "word_length_bucket": word_bucket(len(mixed_tokens)),
                "ai_sentence_count": int(sum(labels)),
                "human_sentence_count": int(len(labels) - sum(labels)),
                "ai_sentence_ratio": (sum(labels) / len(labels)) if labels else 0.0,
                "ai_token_ratio_lir": float(record.get("lir", 0.0) or 0.0),
                "lir_bucket": lir_bucket(float(record.get("lir", 0.0) or 0.0)),
                "jaccard_distance": record.get("jaccard_distance"),
                "jaccard_similarity": None if record.get("jaccard_distance") is None else 1 - float(record["jaccard_distance"]),
                "ngram_cosine_distance": record.get("cosine_distance"),
                "ngram_cosine_similarity": None if record.get("cosine_distance") is None else 1 - float(record["cosine_distance"]),
                "original_ttr": type_token_ratio(original_tokens),
                "mixed_ttr": type_token_ratio(mixed_tokens),
                "ttr_shift": type_token_ratio(mixed_tokens) - type_token_ratio(original_tokens),
                "aligned_sentence_count": aligned,
                "exact_copy_ai_sentence_count": exact_copy_ai_sentences,
                "exact_copy_ai_sentence_ratio": (
                    exact_copy_ai_sentences / len(ai_sentence_indices)
                    if ai_sentence_indices and aligned
                    else None
                ),
                "avg_rewrite_sentence_edit_distance": (
                    sum(rewrite_distances) / len(rewrite_distances) if rewrite_distances else None
                ),
                "rewrite_sentence_observations": len(rewrite_distances),
                "ai_span_count": len(spans),
                "avg_ai_span_length": (sum(spans) / len(spans)) if spans else 0.0,
                "max_ai_span_length": max(spans) if spans else 0,
                "is_human_baseline": rewrite_model == "human" or math.isclose(float(record.get("target_ai_ratio", 0.0) or 0.0), 0.0),
            }
            raw_rows.append(row)

            dedup_key = (record_id, rewrite_model)
            if dedup_key not in seen_keys:
                seen_keys.add(dedup_key)
                dedup_rows.append(row)

    return pd.DataFrame(raw_rows), pd.DataFrame(dedup_rows)


def save_table(df: pd.DataFrame, output_dir: Path, name: str) -> None:
    df.to_csv(output_dir / f"{name}.csv", index=False)
    markdown_df = df.copy()
    for column in markdown_df.columns:
        if pd.api.types.is_float_dtype(markdown_df[column]):
            markdown_df[column] = markdown_df[column].map(lambda value: f"{value:.4f}" if pd.notna(value) else "")
    (output_dir / f"{name}.md").write_text(markdown_df.to_markdown(index=False), encoding="utf-8")


def build_overview_table(records: pd.DataFrame, source_human: pd.DataFrame, input_paths: list[Path]) -> pd.DataFrame:
    non_human = records[~records["is_human_baseline"]].copy()

    metrics = [
        ("input_files", len(input_paths)),
        ("human_source_documents_in_pool", int(source_human["source_doc_id"].nunique()) if not source_human.empty else records["source_doc_id"].nunique()),
        ("human_source_documents_covered", int(records["source_doc_id"].nunique())),
        (
            "source_document_coverage_rate",
            (
                records["source_doc_id"].nunique() / source_human["source_doc_id"].nunique()
                if not source_human.empty and source_human["source_doc_id"].nunique()
                else 1.0
            ),
        ),
        ("mixed_documents", int((~records["is_human_baseline"]).sum())),
        ("human_baseline_documents", int(records["is_human_baseline"].sum())),
        ("total_documents", int(len(records))),
        ("total_sentences", int(records["n_sentences"].sum())),
        ("total_words", int(records["mixed_word_count"].sum())),
        ("average_document_words", float(records["mixed_word_count"].mean())),
        ("average_document_sentences", float(records["n_sentences"].mean())),
        ("average_sentence_words", float(records["mixed_word_count"].sum() / records["n_sentences"].sum())),
        ("number_of_source_human_datasets", int(records["source_dataset"].nunique())),
        ("number_of_source_domains", int(records["source_domain"].nunique())),
        ("number_of_rewriting_llms", int(non_human["rewrite_model"].nunique())),
        ("number_of_llm_families", int(non_human["llm_family"].nunique())),
        ("number_of_mixing_strategies", int(records["mixing_mode"].nunique())),
        ("number_of_ai_ratio_bins", int(records["target_ai_ratio"].nunique())),
        ("mean_length_change_ratio_non_human", safe_mean(non_human["length_change_ratio"])),
        ("mean_jaccard_similarity_non_human", safe_mean(non_human["jaccard_similarity"])),
        ("mean_ngram_cosine_similarity_non_human", safe_mean(non_human["ngram_cosine_similarity"])),
        ("mean_rewrite_edit_distance_non_human", safe_mean(non_human["avg_rewrite_sentence_edit_distance"])),
        ("mean_ai_span_length_non_human", safe_mean(non_human["avg_ai_span_length"])),
        ("mean_exact_copy_ai_sentence_ratio_non_human", safe_mean(non_human["exact_copy_ai_sentence_ratio"])),
    ]

    return pd.DataFrame(metrics, columns=["metric", "value"])


def build_run_summary(records: pd.DataFrame, source_human: pd.DataFrame) -> pd.DataFrame:
    expected_docs = int(source_human["source_doc_id"].nunique()) if not source_human.empty else 0
    rows: list[dict] = []
    for run_name, group in records.groupby("run_name", sort=True):
        non_human = group[~group["is_human_baseline"]]
        covered_docs = int(group["source_doc_id"].nunique())
        rows.append(
            {
                "run_name": run_name,
                "samples": int(len(group)),
                "mixed_samples": int(len(non_human)),
                "covered_source_docs": covered_docs,
                "coverage_rate": (covered_docs / expected_docs) if expected_docs else None,
                "rewrite_models": ", ".join(sorted(set(non_human["rewrite_model"]))) if not non_human.empty else "",
                "mixing_modes": ", ".join(sorted(set(group["mixing_mode"]))),
                "ai_ratio_bins": ", ".join(sorted(set(group["target_ai_ratio_label"]), key=lambda value: int(value[:-1]))),
                "avg_doc_words": float(group["mixed_word_count"].mean()),
                "avg_doc_sentences": float(group["n_sentences"].mean()),
                "avg_length_change_ratio_non_human": safe_mean(non_human["length_change_ratio"]),
                "avg_jaccard_similarity_non_human": safe_mean(non_human["jaccard_similarity"]),
                "avg_ngram_cosine_similarity_non_human": safe_mean(non_human["ngram_cosine_similarity"]),
                "avg_rewrite_edit_distance_non_human": safe_mean(non_human["avg_rewrite_sentence_edit_distance"]),
                "avg_ai_span_length_non_human": safe_mean(non_human["avg_ai_span_length"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["samples", "run_name"], ascending=[False, True])


def build_count_table(records: pd.DataFrame, column: str, value_name: str = "samples") -> pd.DataFrame:
    counts = (
        records[column]
        .value_counts(dropna=False)
        .rename_axis(column)
        .reset_index(name=value_name)
        .sort_values([value_name, column], ascending=[False, True], na_position="last")
    )
    counts["share"] = counts[value_name] / counts[value_name].sum()
    return counts


def build_ratio_by_run_table(records: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        records.pivot_table(
            index="target_ai_ratio_label",
            columns="run_name",
            values="record_id",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
    )
    sort_key = pivot["target_ai_ratio_label"].map(lambda value: int(str(value).rstrip("%")))
    return pivot.iloc[sort_key.argsort()].reset_index(drop=True)


def build_strategy_by_ratio_table(records: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        records.pivot_table(
            index="target_ai_ratio_label",
            columns="mixing_mode",
            values="record_id",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
    )
    sort_key = pivot["target_ai_ratio_label"].map(lambda value: int(str(value).rstrip("%")))
    return pivot.iloc[sort_key.argsort()].reset_index(drop=True)


def build_split_balance_table(records: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        records.groupby("split", dropna=False)
        .agg(
            samples=("record_id", "count"),
            source_docs=("source_doc_id", "nunique"),
            total_sentences=("n_sentences", "sum"),
            ai_sentences=("ai_sentence_count", "sum"),
            human_sentences=("human_sentence_count", "sum"),
        )
        .reset_index()
    )
    grouped["ai_sentence_ratio"] = grouped["ai_sentences"] / grouped["total_sentences"]
    grouped["sample_share"] = grouped["samples"] / grouped["samples"].sum()
    grouped["source_doc_share"] = grouped["source_docs"] / grouped["source_docs"].sum()
    return grouped.sort_values("split")


def build_metric_by_ratio_table(records: pd.DataFrame) -> pd.DataFrame:
    non_human = records[~records["is_human_baseline"]]
    grouped = (
        non_human.groupby("target_ai_ratio_label")
        .agg(
            samples=("record_id", "count"),
            avg_doc_words=("mixed_word_count", "mean"),
            avg_length_change_ratio=("length_change_ratio", "mean"),
            avg_jaccard_similarity=("jaccard_similarity", "mean"),
            avg_ngram_cosine_similarity=("ngram_cosine_similarity", "mean"),
            avg_rewrite_edit_distance=("avg_rewrite_sentence_edit_distance", "mean"),
            avg_ai_span_length=("avg_ai_span_length", "mean"),
            avg_ttr_shift=("ttr_shift", "mean"),
        )
        .reset_index()
    )
    sort_key = grouped["target_ai_ratio_label"].map(lambda value: int(str(value).rstrip("%")))
    return grouped.iloc[sort_key.argsort()].reset_index(drop=True)


def build_metric_by_model_table(records: pd.DataFrame) -> pd.DataFrame:
    non_human = records[~records["is_human_baseline"]]
    grouped = (
        non_human.groupby(["rewrite_model", "llm_family"])
        .agg(
            samples=("record_id", "count"),
            source_docs=("source_doc_id", "nunique"),
            avg_doc_words=("mixed_word_count", "mean"),
            avg_length_change_ratio=("length_change_ratio", "mean"),
            avg_jaccard_similarity=("jaccard_similarity", "mean"),
            avg_ngram_cosine_similarity=("ngram_cosine_similarity", "mean"),
            avg_rewrite_edit_distance=("avg_rewrite_sentence_edit_distance", "mean"),
            avg_ai_span_length=("avg_ai_span_length", "mean"),
            avg_exact_copy_ai_sentence_ratio=("exact_copy_ai_sentence_ratio", "mean"),
        )
        .reset_index()
        .sort_values(["samples", "rewrite_model"], ascending=[False, True])
    )
    return grouped


def plot_bar(table: pd.DataFrame, x: str, y: str, title: str, output_path: Path, dpi: int, xlabel: str = "", ylabel: str = "") -> None:
    plt.figure(figsize=(10, 5))
    plt.bar(table[x].astype(str), table[y], color="#4C78A8")
    plt.title(title)
    plt.xlabel(xlabel or x)
    plt.ylabel(ylabel or y)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def plot_split_balance(table: pd.DataFrame, output_path: Path, dpi: int) -> None:
    plt.figure(figsize=(8, 5))
    x = range(len(table))
    plt.bar(x, table["samples"], color="#59A14F", width=0.6, label="samples")
    plt.plot(x, table["ai_sentence_ratio"] * table["samples"].max(), color="#E15759", marker="o", label="AI sentence ratio (scaled)")
    plt.xticks(list(x), table["split"])
    plt.title("Split Balance")
    plt.ylabel("Samples")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def plot_metric_by_ratio(table: pd.DataFrame, output_path: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    x = table["target_ai_ratio_label"]

    axes[0].plot(x, table["avg_length_change_ratio"], marker="o", color="#4C78A8")
    axes[0].set_title("Length Change Ratio")
    axes[0].set_ylabel("Mixed / Original")

    axes[1].plot(x, table["avg_jaccard_similarity"], marker="o", color="#F28E2B", label="Jaccard")
    axes[1].plot(x, table["avg_ngram_cosine_similarity"], marker="o", color="#59A14F", label="2-gram cosine")
    axes[1].set_title("Similarity by Ratio")
    axes[1].legend()

    axes[2].plot(x, table["avg_rewrite_edit_distance"], marker="o", color="#E15759", label="Edit distance")
    axes[2].plot(x, table["avg_ai_span_length"], marker="o", color="#B07AA1", label="AI span")
    axes[2].set_title("Rewrite Intensity by Ratio")
    axes[2].legend()

    for axis in axes:
        axis.tick_params(axis="x", rotation=30)
        axis.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def plot_ratio_by_run(table: pd.DataFrame, output_path: Path, dpi: int) -> None:
    plot_df = table.set_index("target_ai_ratio_label")
    ax = plot_df.plot(kind="bar", figsize=(12, 6))
    ax.set_title("Samples per AI-Ratio Bin by Run")
    ax.set_xlabel("AI ratio bin")
    ax.set_ylabel("Samples")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def plot_strategy_by_ratio_stacked(table: pd.DataFrame, output_path: Path, dpi: int) -> None:
    plot_df = table.copy()
    x_labels = plot_df["target_ai_ratio_label"].astype(str).tolist()
    block_values = plot_df["block_replace"].tolist() if "block_replace" in plot_df.columns else [0] * len(plot_df)
    random_values = plot_df["random_scatter"].tolist() if "random_scatter" in plot_df.columns else [0] * len(plot_df)
    totals = [block_count + random_count for block_count, random_count in zip(block_values, random_values)]

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 18,
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
        }
    ):
        fig, ax = plt.subplots(figsize=(13.33, 7.5), facecolor="white")
        ax.set_facecolor("#FBFCFE")

        block_color = "#163B65"
        random_color = "#F28E2B"
        edge_color = "#0F172A"

        ax.bar(
            x_labels,
            block_values,
            width=0.68,
            color=block_color,
            edgecolor=edge_color,
            linewidth=1.1,
            label="Block Replace",
            zorder=3,
        )
        ax.bar(
            x_labels,
            random_values,
            width=0.68,
            bottom=block_values,
            color=random_color,
            edgecolor=edge_color,
            linewidth=1.1,
            label="Random Scatter",
            zorder=3,
        )

        max_total = max(totals) if totals else 0
        ax.set_ylim(0, max_total * 1.16 if max_total else 1)

        for idx, (block_count, random_count, total) in enumerate(zip(block_values, random_values, totals)):
            ax.text(
                idx,
                total + max_total * 0.02,
                f"{total:,}",
                ha="center",
                va="bottom",
                fontsize=18,
                fontweight="bold",
                color="#111827",
            )
            if block_count > 0:
                ax.text(
                    idx,
                    block_count / 2,
                    f"{block_count:,}",
                    ha="center",
                    va="center",
                    fontsize=15,
                    fontweight="bold",
                    color="white",
                )
            if random_count > 0:
                ax.text(
                    idx,
                    block_count + random_count / 2,
                    f"{random_count:,}",
                    ha="center",
                    va="center",
                    fontsize=15,
                    fontweight="bold",
                    color="#111827",
                )

        ax.set_xlabel("AI Ratio Bin", fontsize=20, labelpad=12)
        ax.set_ylabel("Number of Samples", fontsize=20, labelpad=12)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):,}"))
        ax.tick_params(axis="x", labelsize=18, width=0, pad=10)
        ax.tick_params(axis="y", labelsize=16)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")

        ax.grid(axis="y", linestyle="--", linewidth=1, alpha=0.25, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.4)
        ax.spines["bottom"].set_linewidth(1.4)

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=2,
            frameon=False,
            fontsize=16,
        )

        fig.tight_layout()
        fig.savefig(output_path, dpi=max(dpi, 300), facecolor="white", bbox_inches="tight")
        fig.savefig(output_path.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
        plt.close(fig)


def write_report(
    output_dir: Path,
    input_paths: list[Path],
    overview: pd.DataFrame,
    run_summary: pd.DataFrame,
    ratio_table: pd.DataFrame,
    model_table: pd.DataFrame,
    strategy_table: pd.DataFrame,
    split_balance: pd.DataFrame,
    metric_by_ratio: pd.DataFrame,
) -> None:
    overview_map = dict(zip(overview["metric"], overview["value"]))
    top_models = model_table.head(6)[["rewrite_model", "samples"]].to_markdown(index=False)
    ratio_markdown = ratio_table.to_markdown(index=False)
    strategy_markdown = strategy_table.to_markdown(index=False)
    run_markdown = run_summary.to_markdown(index=False)
    split_markdown = split_balance.to_markdown(index=False)
    metric_markdown = metric_by_ratio.to_markdown(index=False)

    report = f"""# Dataset Statistics Report

## Scope

- Input files: {len(input_paths)}
- Files analyzed:
{chr(10).join(f"  - {path.relative_to(PROJECT_ROOT)}" for path in input_paths)}

## Overview

- Human source documents in pool: {int(float(overview_map['human_source_documents_in_pool']))}
- Covered source documents: {int(float(overview_map['human_source_documents_covered']))}
- Source coverage rate: {float(overview_map['source_document_coverage_rate']):.2%}
- Mixed documents: {int(float(overview_map['mixed_documents']))}
- Human baseline documents: {int(float(overview_map['human_baseline_documents']))}
- Total documents: {int(float(overview_map['total_documents']))}
- Total sentences: {int(float(overview_map['total_sentences']))}
- Total words: {int(float(overview_map['total_words']))}
- Average document length: {float(overview_map['average_document_words']):.2f} words / {float(overview_map['average_document_sentences']):.2f} sentences
- Average sentence length: {float(overview_map['average_sentence_words']):.2f} words
- Source human datasets: {int(float(overview_map['number_of_source_human_datasets']))}
- Source domains: {int(float(overview_map['number_of_source_domains']))}
- Rewriting LLMs: {int(float(overview_map['number_of_rewriting_llms']))}
- LLM families: {int(float(overview_map['number_of_llm_families']))}
- Mixing strategies: {int(float(overview_map['number_of_mixing_strategies']))}
- AI-ratio bins: {int(float(overview_map['number_of_ai_ratio_bins']))}

## Coverage by Run

{run_markdown}

## Distribution

### Samples per AI-ratio bin

{ratio_markdown}

### Samples per strategy

{strategy_markdown}

### Top rewriting models

{top_models}

## Split Balance

{split_markdown}

## Difficulty Indicators by Ratio

{metric_markdown}

## Generated Assets

- Tables: `*.csv`, `*.md`
- Charts:
  - `samples_per_ratio.png`
  - `samples_per_model.png`
  - `samples_per_strategy.png`
  - `strategy_by_ratio_stacked.png`
  - `samples_per_domain.png`
  - `samples_per_word_bucket.png`
  - `split_balance.png`
  - `metrics_by_ratio.png`
  - `ratio_by_run.png`
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_paths = expand_input_patterns(args.inputs)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_human, split_map = load_source_human_metadata(args.source_human)
    raw_records, records = build_record_rows(input_paths, split_map)

    overview = build_overview_table(records, source_human, input_paths)
    run_summary = build_run_summary(raw_records, source_human)
    ratio_table = build_count_table(records, "target_ai_ratio_label")
    model_table = build_count_table(records[~records["is_human_baseline"]], "rewrite_model")
    family_table = build_count_table(records[~records["is_human_baseline"]], "llm_family")
    strategy_table = build_count_table(records, "mixing_mode")
    domain_table = build_count_table(records, "source_domain")
    dataset_table = build_count_table(records, "source_dataset")
    split_table = build_count_table(records, "split")
    word_bucket_table = build_count_table(records, "word_length_bucket")
    lir_bucket_table = build_count_table(records, "lir_bucket")
    ratio_by_run = build_ratio_by_run_table(raw_records)
    strategy_by_ratio = build_strategy_by_ratio_table(records)
    split_balance = build_split_balance_table(records)
    metric_by_ratio = build_metric_by_ratio_table(records)
    metric_by_model = build_metric_by_model_table(records)

    save_table(overview, args.output_dir, "overview")
    save_table(run_summary, args.output_dir, "run_summary")
    save_table(ratio_table, args.output_dir, "samples_per_ratio")
    save_table(model_table, args.output_dir, "samples_per_model")
    save_table(family_table, args.output_dir, "samples_per_llm_family")
    save_table(strategy_table, args.output_dir, "samples_per_strategy")
    save_table(domain_table, args.output_dir, "samples_per_domain")
    save_table(dataset_table, args.output_dir, "samples_per_source_dataset")
    save_table(split_table, args.output_dir, "samples_per_split")
    save_table(word_bucket_table, args.output_dir, "samples_per_word_bucket")
    save_table(lir_bucket_table, args.output_dir, "samples_per_lir_bucket")
    save_table(ratio_by_run, args.output_dir, "ratio_by_run")
    save_table(strategy_by_ratio, args.output_dir, "strategy_by_ratio")
    save_table(split_balance, args.output_dir, "split_balance")
    save_table(metric_by_ratio, args.output_dir, "metrics_by_ratio")
    save_table(metric_by_model, args.output_dir, "metrics_by_model")

    plot_bar(ratio_table, "target_ai_ratio_label", "samples", "Samples per AI-Ratio Bin", args.output_dir / "samples_per_ratio.png", args.figure_dpi, xlabel="AI ratio bin", ylabel="Samples")
    plot_bar(model_table.head(12), "rewrite_model", "samples", "Samples per Rewriting Model", args.output_dir / "samples_per_model.png", args.figure_dpi, xlabel="Rewriting model", ylabel="Samples")
    plot_bar(strategy_table, "mixing_mode", "samples", "Samples per Mixing Strategy", args.output_dir / "samples_per_strategy.png", args.figure_dpi, xlabel="Mixing strategy", ylabel="Samples")
    plot_strategy_by_ratio_stacked(strategy_by_ratio, args.output_dir / "strategy_by_ratio_stacked.png", args.figure_dpi)
    plot_bar(domain_table, "source_domain", "samples", "Samples per Human Source Domain", args.output_dir / "samples_per_domain.png", args.figure_dpi, xlabel="Source domain", ylabel="Samples")
    plot_bar(word_bucket_table, "word_length_bucket", "samples", "Document Length Distribution", args.output_dir / "samples_per_word_bucket.png", args.figure_dpi, xlabel="Word bucket", ylabel="Samples")
    plot_split_balance(split_balance, args.output_dir / "split_balance.png", args.figure_dpi)
    plot_metric_by_ratio(metric_by_ratio, args.output_dir / "metrics_by_ratio.png", args.figure_dpi)
    plot_ratio_by_run(ratio_by_run, args.output_dir / "ratio_by_run.png", args.figure_dpi)

    write_report(
        output_dir=args.output_dir,
        input_paths=input_paths,
        overview=overview,
        run_summary=run_summary,
        ratio_table=ratio_table,
        model_table=model_table,
        strategy_table=strategy_table,
        split_balance=split_balance,
        metric_by_ratio=metric_by_ratio,
    )

    metadata = {
        "input_files": [str(path.relative_to(PROJECT_ROOT)) for path in input_paths],
        "raw_records": int(len(raw_records)),
        "deduplicated_records": int(len(records)),
        "output_dir": str(args.output_dir.relative_to(PROJECT_ROOT)),
        "source_human_path": str(args.source_human.relative_to(PROJECT_ROOT)) if args.source_human.exists() else None,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved dataset statistics to {args.output_dir}")


if __name__ == "__main__":
    main()
