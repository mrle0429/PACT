"""
已采样人类文本清洗模块。

目标：
- 对采样后的 JSONL 做保守清洗，提升后续分句质量
- 优先移除高置信度的网页/导航噪声，不对自然语言内容做过度裁剪
- 保持输出格式兼容：{"id": "...", "text": "...", "sentence_count": 12}
"""
from __future__ import annotations

import html
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pysbd

from ..utils import get_logger

logger = get_logger(__name__)

try:
    import ftfy  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    ftfy = None


_COMMON_MOJIBAKE_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("\u00a0", " "),
    ("\u200b", ""),
    ("\ufeff", ""),
    ("Â£", "£"),
    ("Â€", "€"),
    ("Â", ""),
    ("â€™", "'"),
    ("â€˜", "'"),
    ('â€œ', '"'),
    ('â€\x9d', '"'),
    ("â€“", "-"),
    ("â€”", "-"),
    ("â€¦", "..."),
)

_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")
_MULTISPACE_RE = re.compile(r"[ \t]{2,}")
_MULTILINE_RE = re.compile(r"\s*\n\s*")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_SOCIAL_HANDLE_RE = re.compile(r"(?:^|\s)(?:@[\w_]+|twitter\.com/\S+|facebook\.com/\S+|instagram\.com/\S+)\b")
_ONLY_PUNCT_OR_SPACE_RE = re.compile(r"^[\W_]+$")
@dataclass
class HumanTextCleanerConfig:
    """清洗器配置。"""

    language: str = "en"
    use_ftfy_if_available: bool = False
    drop_boilerplate_sentences: bool = True
    drop_natural_language_boilerplate: bool = False
    min_sentences_to_keep: int = 1


@dataclass
class RecordCleanResult:
    """单条记录清洗结果。"""

    record: dict[str, Any]
    original_sentence_count: int
    cleaned_sentence_count: int
    changed: bool
    removed_sentences: int = 0
    removal_reasons: Counter[str] = field(default_factory=Counter)
    sentence_logs: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class DatasetCleanStats:
    """整份 JSONL 清洗统计。"""

    total_records: int = 0
    written_records: int = 0
    skipped_records: int = 0
    changed_records: int = 0
    total_removed_sentences: int = 0
    removal_reasons: Counter[str] = field(default_factory=Counter)
    logged_records: int = 0


class HumanTextCleaner:
    """对已采样人类文本做保守清洗。"""

    _HARD_BOILERPLATE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
        ("source_credit", re.compile(r"^(?:Source|Photo credit):\s*(?:https?://\S+|www\.\S+)", re.IGNORECASE)),
        ("reporting_credit", re.compile(r"^\(?Reporting by .+; Editing by .+\)?$", re.IGNORECASE)),
        ("link_only", re.compile(r"^(?:https?://\S+|www\.\S+)$", re.IGNORECASE)),
        ("email_social_only", re.compile(r"^(?:[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|twitter\.com/\S+|facebook\.com/\S+|instagram\.com/\S+)(?:\s+(?:@[\w_]+|twitter\.com/\S+|facebook\.com/\S+|instagram\.com/\S+))*$", re.IGNORECASE)),
        ("orphan_heading", re.compile(r"^(?:Latest [A-Za-z ]+ updates|How to help with [A-Za-z ]+|The other accused men are:)$", re.IGNORECASE)),
    )
    _SOFT_BOILERPLATE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
        ("promo_nav", re.compile(r"^(?:Editor's Picks\b|More on this story\b|DON['’]T MISS\b|Trending right now\b)", re.IGNORECASE)),
        ("follow_social", re.compile(r"^(?:Follow @\w+|BBCNewsbeat on \w+|See the original version of this article\b)", re.IGNORECASE)),
        ("promo_game", re.compile(r"^Take part in our new\b", re.IGNORECASE)),
    )

    def __init__(self, cfg: HumanTextCleanerConfig | None = None):
        self.cfg = cfg or HumanTextCleanerConfig()
        # 保持句子文本表面形式，避免 clean=True 改写引号/撇号。
        self._segmenter = pysbd.Segmenter(language=self.cfg.language, clean=False)

    def clean_dataset(
        self,
        input_path: str | Path,
        output_path: str | Path,
        log_path: str | Path | None = None,
        summary_path: str | Path | None = None,
    ) -> DatasetCleanStats:
        """清洗整份 JSONL 并写出新文件。"""
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = None
        summary_target = Path(summary_path) if summary_path is not None else None
        if log_path is not None:
            log_target = Path(log_path)
            log_target.parent.mkdir(parents=True, exist_ok=True)
            log_file = log_target.open("w", encoding="utf-8")
        if summary_target is not None:
            summary_target.parent.mkdir(parents=True, exist_ok=True)

        stats = DatasetCleanStats()
        try:
            with input_path.open(encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
                for lineno, line in enumerate(src, 1):
                    line = line.strip()
                    if not line:
                        continue
                    stats.total_records += 1
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        logger.warning(f"第 {lineno} 行 JSON 解析失败，已跳过: {exc}")
                        stats.skipped_records += 1
                        continue

                    result = self.clean_record(record)
                    skipped = result.cleaned_sentence_count < self.cfg.min_sentences_to_keep
                    if skipped:
                        stats.skipped_records += 1
                    else:
                        if result.changed:
                            stats.changed_records += 1
                        stats.written_records += 1
                        stats.total_removed_sentences += result.removed_sentences
                        stats.removal_reasons.update(result.removal_reasons)
                        dst.write(json.dumps(result.record, ensure_ascii=False) + "\n")

                    if log_file is not None:
                        stats.logged_records += 1
                        log_file.write(
                            json.dumps(
                                self._build_log_entry(record, result, lineno, skipped),
                                ensure_ascii=False,
                            ) + "\n"
                        )
        finally:
            if log_file is not None:
                log_file.close()

        logger.info(
            "清洗完成: 输入 %s 条, 写出 %s 条, 修改 %s 条, 删除句子 %s 条, 日志 %s 条",
            stats.total_records,
            stats.written_records,
            stats.changed_records,
            stats.total_removed_sentences,
            stats.logged_records,
        )
        if summary_target is not None:
            summary_target.write_text(
                json.dumps(self._build_summary(stats), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return stats

    def clean_record(self, record: dict[str, Any]) -> RecordCleanResult:
        """清洗单条记录。"""
        original_text = str(record.get("text", "") or "")
        doc_id = str(record.get("id", ""))
        source = doc_id.split("_")[0] if "_" in doc_id else ""
        normalized = self._normalize_text(original_text)

        original_sentences = self._split_sentences(original_text)
        candidate_sentences = self._split_sentences(normalized)

        kept_sentences: list[str] = []
        removal_reasons: Counter[str] = Counter()
        removed_sentences = 0
        sentence_logs: list[dict[str, Any]] = []

        for sentence in candidate_sentences:
            original_sentence = sentence.strip()
            prepared_sentence = sentence.strip()
            reason = self._classify_sentence_noise(prepared_sentence, source)
            if reason is not None and self.cfg.drop_boilerplate_sentences:
                removal_reasons[reason] += 1
                removed_sentences += 1
                sentence_logs.append({
                    "original_sentence": original_sentence,
                    "prepared_sentence": prepared_sentence,
                    "action": "removed",
                    "reason": reason,
                })
                continue
            kept_sentences.append(prepared_sentence)
            sentence_logs.append({
                "original_sentence": original_sentence,
                "prepared_sentence": prepared_sentence,
                "action": "kept",
                "reason": None,
            })

        cleaned_text = self._finalize_text(" ".join(kept_sentences))
        cleaned_sentences = self._split_sentences(cleaned_text)

        cleaned_record = dict(record)
        cleaned_record["text"] = cleaned_text
        cleaned_record["sentence_count"] = len(cleaned_sentences)

        changed = (
            cleaned_text != original_text
            or len(cleaned_sentences) != record.get("sentence_count", 0)
        )
        return RecordCleanResult(
            record=cleaned_record,
            original_sentence_count=len(original_sentences),
            cleaned_sentence_count=len(cleaned_sentences),
            changed=changed,
            removed_sentences=removed_sentences,
            removal_reasons=removal_reasons,
            sentence_logs=sentence_logs,
        )

    def _normalize_text(self, text: str) -> str:
        text = text.strip()
        if not text:
            return ""

        if self.cfg.use_ftfy_if_available and ftfy is not None:
            text = ftfy.fix_text(text)

        text = html.unescape(text)

        for src, dst in _COMMON_MOJIBAKE_REPLACEMENTS:
            text = text.replace(src, dst)

        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("\\n", "\n").replace("\\r", "\n")
        text = _HTML_TAG_RE.sub("", text)
        text = _MULTILINE_RE.sub(" ", text)
        text = _MULTISPACE_RE.sub(" ", text)
        return text.strip()

    def _finalize_text(self, text: str) -> str:
        text = _MULTISPACE_RE.sub(" ", text)
        return text.strip()

    def _split_sentences(self, text: str) -> list[str]:
        if not text:
            return []
        return [s.strip() for s in self._segmenter.segment(text) if s.strip()]

    def _classify_sentence_noise(self, sentence: str, source: str) -> str | None:
        sentence = sentence.strip()
        if not sentence:
            return "empty"
        if _ONLY_PUNCT_OR_SPACE_RE.match(sentence):
            return "punct_only"

        for reason, pattern in self._HARD_BOILERPLATE_PATTERNS:
            if pattern.search(sentence):
                return reason

        if self.cfg.drop_natural_language_boilerplate:
            for reason, pattern in self._SOFT_BOILERPLATE_PATTERNS:
                if pattern.search(sentence):
                    return reason

        if _URL_RE.fullmatch(sentence):
            return "link_only"

        if _EMAIL_RE.fullmatch(sentence):
            return "email_social_only"

        if self._is_contact_tail(sentence):
            return "email_social_only"

        if self._is_source_specific_noise(sentence, source):
            return "source_specific"

        return None

    def _is_contact_tail(self, sentence: str) -> bool:
        compact = sentence.strip()
        if not compact:
            return False
        has_email = bool(_EMAIL_RE.search(compact))
        has_social = bool(_SOCIAL_HANDLE_RE.search(compact))
        # 典型尾巴: "name@example.com twitter.com/foo"
        return has_email and has_social

    def _is_source_specific_noise(self, sentence: str, source: str) -> bool:
        if source in {"owt", "xsum"}:
            if sentence.endswith(":") and len(sentence.split()) <= 8:
                return True
        return False

    def _build_log_entry(
        self,
        original_record: dict[str, Any],
        result: RecordCleanResult,
        lineno: int,
        skipped: bool,
    ) -> dict[str, Any]:
        doc_id = str(original_record.get("id", f"line_{lineno}"))
        source = doc_id.split("_")[0] if "_" in doc_id else ""
        removed_details = [item for item in result.sentence_logs if item["action"] == "removed"]
        changed_but_kept = [
            item for item in result.sentence_logs
            if item["action"] == "kept" and item["original_sentence"] != item["prepared_sentence"]
        ]
        return {
            "line_number": lineno,
            "id": doc_id,
            "source": source,
            "status": "skipped" if skipped else "written",
            "changed": result.changed,
            "original_sentence_count": result.original_sentence_count,
            "cleaned_sentence_count": result.cleaned_sentence_count,
            "removed_sentence_count": result.removed_sentences,
            "removal_reasons": dict(result.removal_reasons),
            "original_text": str(original_record.get("text", "") or ""),
            "cleaned_text": result.record["text"],
            "removed_sentence_details": removed_details,
            "changed_kept_sentence_details": changed_but_kept,
        }

    def _build_summary(self, stats: DatasetCleanStats) -> dict[str, Any]:
        return {
            "total_records": stats.total_records,
            "written_records": stats.written_records,
            "skipped_records": stats.skipped_records,
            "changed_records": stats.changed_records,
            "total_removed_sentences": stats.total_removed_sentences,
            "logged_records": stats.logged_records,
            "removal_reasons": dict(stats.removal_reasons),
        }
