"""
Single-text web demo service for the PACT construction flow.

The public web API needs a slightly different contract than the batch pipeline:
it should always be able to show local intermediate steps, even when an LLM call
times out or fails. This module keeps that behavior separate from the dataset
writer-oriented pipeline code.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import random
from dataclasses import asdict, dataclass
from typing import Any, Literal

from .config import DatasetConfig, SUPPORTED_MODELS
from .label_calculator import (
    compute_cosine_distance,
    compute_jaccard_distance,
    compute_labels,
    compute_sentence_jaccard,
)
from .llm_rewriter import RewriteResult, create_rewriter
from .sentence_processor import (
    MixingMode,
    SentenceSelection,
    create_sentence_selection,
    split_into_sentences,
)

MAX_DEMO_TEXT_CHARS = 6000
MIN_DEMO_SENTENCES = 2
DEFAULT_DEMO_TIMEOUT_SECONDS = 60.0
_configured_allowlist = tuple(
    name.strip()
    for name in os.getenv("PACT_DEMO_MODELS", "").split(",")
    if name.strip()
)
DEMO_MODEL_ALLOWLIST = tuple(
    name for name in (_configured_allowlist or tuple(SUPPORTED_MODELS.keys()))
    if name in SUPPORTED_MODELS
)
DEMO_RATIOS = (0.2, 0.4, 0.6, 0.8, 1.0)
DEMO_MIXING_MODES: tuple[MixingMode, ...] = ("block_replace", "random_scatter")


class DemoValidationError(ValueError):
    """Raised when a demo request is invalid before any provider call."""


class DemoRewriteError(RuntimeError):
    """Raised when the provider call fails after local steps are available."""


@dataclass(frozen=True)
class SampleText:
    id: str
    title: str
    category: Literal["academic", "news", "web", "essay"]
    text: str


SAMPLE_TEXTS: tuple[SampleText, ...] = (
    SampleText(
        id="academic-graph-learning",
        title="Graph Learning Abstract",
        category="academic",
        text=(
            "Graph neural networks have become a common approach for learning from relational data. "
            "However, many benchmarks contain structural shortcuts that make evaluation less reliable. "
            "We introduce a controlled protocol that separates neighborhood memorization from genuine generalization. "
            "The protocol perturbs graph topology while preserving label distributions and feature statistics. "
            "Experiments on citation and molecular datasets show that several popular architectures lose accuracy under this setting. "
            "These findings suggest that future benchmarks should report robustness across multiple graph construction regimes."
        ),
    ),
    SampleText(
        id="news-city-transit",
        title="City Transit Report",
        category="news",
        text=(
            "City officials announced a new transit schedule on Monday after months of public consultation. "
            "The revised plan adds more frequent buses during the morning commute and extends evening service on three major routes. "
            "Local business owners said the changes could help workers who currently rely on expensive ride-hailing trips. "
            "Some residents remain concerned that weekend coverage is still too limited for neighborhoods far from the rail line. "
            "The transportation department said it will review passenger data after the first ninety days. "
            "A second round of adjustments is expected before the end of the year."
        ),
    ),
    SampleText(
        id="web-productivity",
        title="Productivity Blog",
        category="web",
        text=(
            "Most productivity advice fails because it treats every task as if it requires the same kind of attention. "
            "A better approach is to separate planning work from execution work before the day begins. "
            "When the next action is already defined, it becomes easier to start without negotiating with yourself. "
            "Short checklists can also reduce the mental load of repeated routines. "
            "The goal is not to fill every hour with activity, but to make important work easier to resume. "
            "Small systems are often more durable than ambitious schedules that collapse after one interruption."
        ),
    ),
    SampleText(
        id="essay-online-learning",
        title="Student Essay",
        category="essay",
        text=(
            "Online learning gives students more control over when and where they study. "
            "This flexibility can be especially helpful for people who have jobs, family responsibilities, or long commutes. "
            "At the same time, virtual classes require a high level of self-discipline. "
            "Students may fall behind if they do not manage deadlines carefully or ask questions when they are confused. "
            "In my view, online learning works best when schools combine recorded material with regular live discussion. "
            "That balance preserves flexibility while still giving students a sense of structure and community."
        ),
    ),
)


@dataclass
class SentencePair:
    index: int
    original: str
    rewritten: str
    selected: bool
    label: int


@dataclass
class DemoStep:
    id: str
    label: str
    status: Literal["complete", "failed"]
    detail: str


@dataclass
class DemoResult:
    ok: bool
    error: str
    original_text: str
    mixed_text: str
    sentences: list[str]
    mixed_sentences: list[str]
    selected_indices: list[int]
    rewrites: dict[int, str]
    sentence_labels: list[int]
    sentence_pairs: list[SentencePair]
    labels: dict[str, Any]
    model: str
    provider_model_id: str
    mixing_mode: str
    target_ratio: float
    input_tokens: int
    output_tokens: int
    steps: list[DemoStep]
    dataset_record: dict[str, Any]


def list_sample_texts() -> list[dict[str, str]]:
    return [asdict(sample) for sample in SAMPLE_TEXTS]


def build_demo_config() -> dict[str, Any]:
    return {
        "models": [
            {
                "name": name,
                "provider": model_cfg.provider,
                "model_id": model_cfg.model_id,
                "temperature": model_cfg.temperature,
                "requests_per_minute": model_cfg.requests_per_minute,
                "max_output_tokens": model_cfg.max_output_tokens,
            }
            for name, model_cfg in SUPPORTED_MODELS.items()
            if name in DEMO_MODEL_ALLOWLIST
        ],
        "mixing_modes": list(DEMO_MIXING_MODES),
        "ratios": list(DEMO_RATIOS),
        "max_text_chars": MAX_DEMO_TEXT_CHARS,
        "default_model": _default_demo_model(),
    }


def validate_demo_request(
    *,
    text: str,
    target_ratio: float,
    mixing_mode: str,
    model: str,
) -> str:
    normalized_text = " ".join(text.split())
    if not normalized_text:
        raise DemoValidationError("请输入文本，或选择一个 sample text。")
    if len(normalized_text) > MAX_DEMO_TEXT_CHARS:
        raise DemoValidationError(
            f"输入文本过长，最多允许 {MAX_DEMO_TEXT_CHARS} 个字符。"
        )
    if model not in DEMO_MODEL_ALLOWLIST:
        raise DemoValidationError("该模型未开放给公开 Demo 使用。")
    if mixing_mode not in DEMO_MIXING_MODES:
        raise DemoValidationError("未知改写模式。")
    if target_ratio not in DEMO_RATIOS:
        raise DemoValidationError("未知 AI 句子比例。")
    return normalized_text


def _default_demo_model() -> str:
    if "MiniMax-M2.7" in DEMO_MODEL_ALLOWLIST:
        return "MiniMax-M2.7"
    if not DEMO_MODEL_ALLOWLIST:
        raise DemoValidationError(
            "PACT_DEMO_MODELS 没有包含任何受支持的模型，请检查服务端配置。"
        )
    return DEMO_MODEL_ALLOWLIST[0]


def build_local_preview(
    *,
    text: str,
    target_ratio: float,
    mixing_mode: str,
    model: str,
    seed: int,
) -> tuple[DatasetConfig, SentenceSelection, list[DemoStep]]:
    normalized_text = validate_demo_request(
        text=text,
        target_ratio=target_ratio,
        mixing_mode=mixing_mode,
        model=model,
    )
    sentences = split_into_sentences(normalized_text)
    if len(sentences) < MIN_DEMO_SENTENCES:
        raise DemoValidationError(
            f"至少需要 {MIN_DEMO_SENTENCES} 个句子才能展示句子级混合构造。"
        )

    cfg = DatasetConfig(
        rewrite_model=model,
        random_seed=seed,
        concurrent_requests=1,
    )
    selection = create_sentence_selection(
        sentences,
        target_ratio,
        mixing_mode,  # type: ignore[arg-type]
        random.Random(seed),
    )
    steps = [
        DemoStep(
            id="split",
            label="Split into sentences",
            status="complete",
            detail=f"{len(sentences)} sentences detected with PySBD.",
        ),
        DemoStep(
            id="select",
            label="Select rewrite targets",
            status="complete",
            detail=(
                f"{len(selection.selected_indices)} sentence(s) selected by "
                f"{mixing_mode} at {target_ratio:.0%} target ratio."
            ),
        ),
    ]
    return cfg, selection, steps


async def run_demo_rewrite(
    *,
    text: str,
    target_ratio: float,
    mixing_mode: str,
    model: str,
    seed: int = 42,
    language_hint: str = "English",
    timeout_seconds: float = DEFAULT_DEMO_TIMEOUT_SECONDS,
) -> DemoResult:
    cfg, selection, steps = build_local_preview(
        text=text,
        target_ratio=target_ratio,
        mixing_mode=mixing_mode,
        model=model,
        seed=seed,
    )

    result = RewriteResult({}, cfg.get_model_config().model_id)
    error = ""
    ok = True
    if selection.selected_indices:
        rewriter = create_rewriter(cfg)
        try:
            result = await asyncio.wait_for(
                rewriter.rewrite(
                    selection.sentences,
                    selection.selected_indices,
                    language_hint=language_hint,
                    task_id=_make_demo_record_id(text, target_ratio, mixing_mode),
                ),
                timeout=timeout_seconds,
            )
            _validate_rewrite_result(selection, result)
            steps.append(
                DemoStep(
                    id="rewrite",
                    label="Rewrite selected sentences",
                    status="complete",
                    detail=(
                        f"{len(result.rewrites)} sentence(s) rewritten by "
                        f"{cfg.rewrite_model}."
                    ),
                )
            )
        except Exception as exc:
            ok = False
            error = f"{type(exc).__name__}: {exc}"
            result = RewriteResult({}, cfg.get_model_config().model_id)
            steps.append(
                DemoStep(
                    id="rewrite",
                    label="Rewrite selected sentences",
                    status="failed",
                    detail=error,
                )
            )
        finally:
            await rewriter.aclose()
    else:
        steps.append(
            DemoStep(
                id="rewrite",
                label="Rewrite selected sentences",
                status="complete",
                detail="No sentence was selected for rewriting.",
            )
        )

    return build_demo_result(
        cfg=cfg,
        selection=selection,
        result=result,
        steps=steps,
        ok=ok,
        error=error,
    )


def build_demo_result(
    *,
    cfg: DatasetConfig,
    selection: SentenceSelection,
    result: RewriteResult,
    steps: list[DemoStep],
    ok: bool,
    error: str,
) -> DemoResult:
    rewrites = result.rewrites if ok else {}
    mixed_sentences = selection.build_mixed_sentences(rewrites)
    original_text = " ".join(selection.sentences)
    labels, label_error = _compute_demo_labels(
        original_text=original_text,
        mixed_sentences=mixed_sentences,
        ai_indices=list(rewrites.keys()),
        cfg=cfg,
    )
    sentence_labels = selection.sentence_label_array(rewrites)
    mixed_text = " ".join(mixed_sentences)
    steps.extend(
        [
            DemoStep(
                id="backfill",
                label="Backfill mixed text",
                status="complete",
                detail="Selected sentences are replaced; unselected sentences stay unchanged.",
            ),
            DemoStep(
                id="labels",
                label="Compute labels",
                status="complete",
                detail="Sentence labels and continuous document labels are computed.",
            ),
        ]
    )

    pairs = [
        SentencePair(
            index=index,
            original=sentence,
            rewritten=mixed_sentences[index],
            selected=index in selection.selected_indices,
            label=sentence_labels[index],
        )
        for index, sentence in enumerate(selection.sentences)
    ]
    dataset_record = {
        "id": _make_demo_record_id(
            " ".join(selection.sentences),
            selection.target_ratio,
            selection.mode,
        ),
        "source_dataset": "web_demo",
        "source_domain": "interactive",
        "original_text": original_text,
        "mixed_text": mixed_text,
        "n_sentences": len(selection.sentences),
        "target_ai_ratio": selection.target_ratio,
        "mixing_mode": selection.mode,
        "rewrite_model": cfg.rewrite_model if ok else "failed",
        "sentence_labels": sentence_labels,
        "lir": labels.get("lir"),
        "jaccard_distance": labels.get("jaccard_distance"),
        "sentence_jaccard": labels.get("sentence_jaccard"),
        "cosine_distance": labels.get("cosine_distance"),
        "extra": {
            "selected_indices": selection.selected_indices,
            "provider_model_id": result.model_id,
            "error": error,
            "label_error": label_error,
        },
    }
    return DemoResult(
        ok=ok,
        error=error,
        original_text=original_text,
        mixed_text=mixed_text,
        sentences=selection.sentences,
        mixed_sentences=mixed_sentences,
        selected_indices=selection.selected_indices,
        rewrites=rewrites,
        sentence_labels=sentence_labels,
        sentence_pairs=pairs,
        labels=labels,
        model=cfg.rewrite_model,
        provider_model_id=result.model_id,
        mixing_mode=selection.mode,
        target_ratio=selection.target_ratio,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        steps=steps,
        dataset_record=dataset_record,
    )


def _compute_demo_labels(
    *,
    original_text: str,
    mixed_sentences: list[str],
    ai_indices: list[int],
    cfg: DatasetConfig,
) -> tuple[dict[str, Any], str]:
    try:
        return compute_labels(
            original_text=original_text,
            mixed_sentences=mixed_sentences,
            ai_indices=ai_indices,
            cfg=cfg,
        ), ""
    except RuntimeError as exc:
        mixed_text = " ".join(mixed_sentences)
        sentence_labels = [
            1 if index in set(ai_indices) else 0
            for index in range(len(mixed_sentences))
        ]
        return {
            "lir": None,
            "jaccard_distance": compute_jaccard_distance(original_text, mixed_text),
            "sentence_jaccard": compute_sentence_jaccard(
                original_text,
                mixed_text,
                sentence_labels,
            ),
            "cosine_distance": compute_cosine_distance(
                original_text,
                mixed_text,
                cfg.ngram_n,
            ),
        }, str(exc)


def _validate_rewrite_result(selection: SentenceSelection, result: RewriteResult) -> None:
    if result.missing_indices or result.invalid_indices or result.extra_indices:
        raise DemoRewriteError(
            "Invalid rewrite response "
            f"(missing={result.missing_indices}, "
            f"invalid={result.invalid_indices}, extra={result.extra_indices})."
        )
    if len(result.rewrites) < len(selection.selected_indices):
        raise DemoRewriteError(
            f"Incomplete rewrite response: expected {len(selection.selected_indices)}, "
            f"got {len(result.rewrites)}."
        )
    after_count = len(split_into_sentences(selection.build_mixed_text(result.rewrites)))
    if after_count != selection.n:
        raise DemoRewriteError(
            f"Sentence count changed after backfill: before={selection.n}, after={after_count}."
        )


def _make_demo_record_id(text: str, target_ratio: float, mixing_mode: str) -> str:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:10]
    return f"demo_{digest}_r{int(target_ratio * 100)}_{mixing_mode}"
