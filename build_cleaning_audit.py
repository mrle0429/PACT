#!/usr/bin/env python3
"""
基于 human_text_cleaner 的 cleaning log 生成审计表。

输出：
- 记录级 CSV：每条文档一行，包含主要变更类型
- 删除句子 CSV：每个被删句子一行
- Markdown 汇总：高层统计与说明
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


HTML_ENTITIES = ("&nbsp;", "&#39;", "&amp;", "&quot;", "&lt;", "&gt;")
MOJIBAKE_MARKERS = ("Â£", "Ã", "â€™", "â€œ", "â€", "\xa0")


def main() -> None:
    parser = argparse.ArgumentParser(description="为 cleaning log 生成详细审计表")
    parser.add_argument(
        "--log-input",
        type=str,
        required=True,
        help="cleaning_log.jsonl 路径",
    )
    parser.add_argument(
        "--record-csv",
        type=str,
        default=None,
        help="记录级审计 CSV 输出路径",
    )
    parser.add_argument(
        "--removed-csv",
        type=str,
        default=None,
        help="删除句子明细 CSV 输出路径",
    )
    parser.add_argument(
        "--summary-md",
        type=str,
        default=None,
        help="Markdown 汇总输出路径",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default=None,
        help="JSON 汇总输出路径",
    )
    args = parser.parse_args()

    log_input = Path(args.log_input)
    if not log_input.exists():
        raise FileNotFoundError(f"cleaning log 不存在: {log_input}")

    stem = _strip_jsonl_suffix(log_input.name)
    parent = log_input.parent
    record_csv = Path(args.record_csv) if args.record_csv else parent / f"{stem}.audit.records.csv"
    removed_csv = Path(args.removed_csv) if args.removed_csv else parent / f"{stem}.audit.removed_sentences.csv"
    summary_md = Path(args.summary_md) if args.summary_md else parent / f"{stem}.audit.md"
    summary_json = Path(args.summary_json) if args.summary_json else parent / f"{stem}.audit.summary.json"

    rows = [json.loads(line) for line in log_input.open(encoding="utf-8") if line.strip()]
    record_rows = [_build_record_row(row) for row in rows]
    removed_rows = _build_removed_rows(rows)

    _write_csv(record_csv, record_rows)
    _write_csv(removed_csv, removed_rows)

    summary_payload = _build_summary(rows, record_rows, removed_rows)
    summary_json.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_md.write_text(_render_summary_md(summary_payload, log_input, record_csv, removed_csv), encoding="utf-8")

    print(json.dumps({
        "log_input": str(log_input),
        "record_csv": str(record_csv),
        "removed_csv": str(removed_csv),
        "summary_md": str(summary_md),
        "summary_json": str(summary_json),
        "total_log_rows": len(rows),
        "changed_records": summary_payload["changed_records"],
        "removed_sentence_rows": len(removed_rows),
    }, ensure_ascii=False, indent=2))


def _strip_jsonl_suffix(name: str) -> str:
    if name.endswith(".jsonl"):
        return name[:-6]
    return name


def _build_record_row(row: dict[str, Any]) -> dict[str, Any]:
    original_text = row.get("original_text", "") or ""
    cleaned_text = row.get("cleaned_text", "") or ""
    removed_sentence_count = int(row.get("removed_sentence_count", 0) or 0)
    removed_reasons = row.get("removal_reasons", {}) or {}

    return {
        "line_number": row.get("line_number", 0),
        "id": row.get("id", ""),
        "source": row.get("source", ""),
        "status": row.get("status", ""),
        "changed": row.get("changed", False),
        "original_sentence_count": row.get("original_sentence_count", 0),
        "cleaned_sentence_count": row.get("cleaned_sentence_count", 0),
        "removed_sentence_count": removed_sentence_count,
        "removed_reasons": _encode_compact_dict(removed_reasons),
        "html_entity_decoded": _has_html_entity_decoded(original_text, cleaned_text),
        "html_tag_removed": _has_html_tag_removed(original_text, cleaned_text),
        "mojibake_fixed": _has_mojibake_fixed(original_text, cleaned_text),
        "escaped_newline_normalized": _has_escaped_newline_normalized(original_text, cleaned_text),
        "sentence_removed": removed_sentence_count > 0,
        "char_length_delta": len(cleaned_text) - len(original_text),
        "original_text_preview": _preview(original_text),
        "cleaned_text_preview": _preview(cleaned_text),
    }


def _build_removed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    removed_rows: list[dict[str, Any]] = []
    for row in rows:
        for item in row.get("removed_sentence_details", []) or []:
            removed_rows.append({
                "line_number": row.get("line_number", 0),
                "id": row.get("id", ""),
                "source": row.get("source", ""),
                "reason": item.get("reason", ""),
                "original_sentence": item.get("original_sentence", ""),
                "prepared_sentence": item.get("prepared_sentence", ""),
            })
    return removed_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_summary(
    rows: list[dict[str, Any]],
    record_rows: list[dict[str, Any]],
    removed_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    changed_records = [r for r in record_rows if r["changed"]]
    by_source = Counter(r["source"] for r in record_rows)
    changed_by_source = Counter(r["source"] for r in changed_records)
    removal_reasons = Counter(r["reason"] for r in removed_rows)
    change_types = Counter()

    for row in changed_records:
        for flag in (
            "html_entity_decoded",
            "html_tag_removed",
            "mojibake_fixed",
            "escaped_newline_normalized",
            "sentence_removed",
        ):
            if row[flag]:
                change_types[flag] += 1

    return {
        "total_records": len(record_rows),
        "changed_records": len(changed_records),
        "unchanged_records": len(record_rows) - len(changed_records),
        "removed_sentence_rows": len(removed_rows),
        "records_by_source": dict(by_source),
        "changed_records_by_source": dict(changed_by_source),
        "change_types": dict(change_types),
        "removal_reasons": dict(removal_reasons),
        "sample_changed_ids": [r["id"] for r in changed_records[:10]],
    }


def _render_summary_md(
    summary: dict[str, Any],
    log_input: Path,
    record_csv: Path,
    removed_csv: Path,
) -> str:
    lines = [
        "# Cleaning Audit",
        "",
        f"- log input: `{log_input}`",
        f"- record audit csv: `{record_csv}`",
        f"- removed sentences csv: `{removed_csv}`",
        "",
        "## Overview",
        "",
        f"- total records: {summary['total_records']}",
        f"- changed records: {summary['changed_records']}",
        f"- unchanged records: {summary['unchanged_records']}",
        f"- removed sentence rows: {summary['removed_sentence_rows']}",
        "",
        "## Records By Source",
        "",
    ]
    for source, count in summary["records_by_source"].items():
        changed = summary["changed_records_by_source"].get(source, 0)
        lines.append(f"- {source}: {count} total, {changed} changed")

    lines.extend([
        "",
        "## Change Types",
        "",
    ])
    for change_type, count in summary["change_types"].items():
        lines.append(f"- {change_type}: {count}")

    lines.extend([
        "",
        "## Removal Reasons",
        "",
    ])
    if summary["removal_reasons"]:
        for reason, count in summary["removal_reasons"].items():
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- none")

    lines.extend([
        "",
        "## Sample Changed IDs",
        "",
    ])
    for doc_id in summary["sample_changed_ids"]:
        lines.append(f"- {doc_id}")

    lines.append("")
    return "\n".join(lines)


def _encode_compact_dict(value: dict[str, Any]) -> str:
    if not value:
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _preview(text: str, limit: int = 220) -> str:
    text = text.replace("\n", "\\n")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _has_html_entity_decoded(original_text: str, cleaned_text: str) -> bool:
    return any(token in original_text for token in HTML_ENTITIES) and not any(
        token in cleaned_text for token in HTML_ENTITIES
    )


def _has_html_tag_removed(original_text: str, cleaned_text: str) -> bool:
    return ("<" in original_text and ">" in original_text) and ("<" not in cleaned_text and ">" not in cleaned_text)


def _has_mojibake_fixed(original_text: str, cleaned_text: str) -> bool:
    return any(token in original_text for token in MOJIBAKE_MARKERS) and not any(
        token in cleaned_text for token in MOJIBAKE_MARKERS
    )


def _has_escaped_newline_normalized(original_text: str, cleaned_text: str) -> bool:
    return ("\\n" in original_text or "\\r" in original_text) and ("\\n" not in cleaned_text and "\\r" not in cleaned_text)


if __name__ == "__main__":
    main()
