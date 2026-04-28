"""
句子级二次 humanize 组件。

职责：
- 构建针对 AI 句子的二次改写 prompt
- 复用现有 provider / model 调用能力
- 复用现有 JSON 解析与单句级校验逻辑
"""
from __future__ import annotations

from typing import Any

from .config import DatasetConfig
from .rewriters import ApiCallLogger, RewriteResult, create_rewriter


def _build_numbered_context(sentences: list[str]) -> str:
    return "\n".join(f"[{index + 1}] {sentence}" for index, sentence in enumerate(sentences))


def build_humanize_system_prompt(language_hint: str = "English") -> str:
    return f"""You are a professional humanizer editor responsible for making AI-written sentences read more like natural human-authored text and not be detected as AI-generated.

Your task is to humanize only the sentences explicitly selected by the user.

Requirements:
- Preserve the original meaning, factual content, entities, tense, and stance.
- Keep the writing natural, fluent, and human-authored in {language_hint}.
- Avoid generic, overly polished, template-like phrasing when a more natural wording is possible.
- Keep roughly similar information density and approximate length.
- Rewrite only the selected sentences.
- Each rewritten value must remain exactly one sentence.
- Do not split, merge, drop, reorder, or summarize sentences.
- Return only a strict JSON object that maps 1-indexed sentence numbers to rewritten sentence text.
- Do not include any explanation, commentary, markdown, or extra text."""


def build_humanize_user_prompt(
    sentences: list[str],
    selected_indices: list[int],
) -> str:
    numbered_context = _build_numbered_context(sentences)
    target_indices_str = ", ".join(str(index + 1) for index in selected_indices)
    example_keys = ", ".join(
        f'"{index + 1}": "<humanized sentence>"'
        for index in selected_indices[:2]
    )

    return f"""Humanize ONLY the following sentence numbers: {target_indices_str}

These selected sentences were previously rewritten by an AI model. Rewrite them so they read like natural human-authored sentences while preserving the original sentence boundaries and meaning.

Article context:
<context>
{numbered_context}
</context>

Return format:
{{{example_keys}, ...}}

Important:
- Return every requested sentence number exactly once.
- Each value must be a single rewritten sentence, not multiple sentences.
- Do not return unrequested sentence numbers."""


class SentenceHumanizer:
    """
    句子级 selective humanizer。

    说明：
    - 复用底层 BaseLLMRewriter 的 provider 实现
    - 复用底层的 `_call_api_with_retry` 与 `_parse_response`
      以确保和主 pipeline 的校验标准一致
    """

    def __init__(self, cfg: DatasetConfig, api_logger: ApiCallLogger):
        self._rewriter = create_rewriter(cfg, api_logger=api_logger)
        self._api_logger = api_logger
        self._model_id = cfg.get_model_config().model_id

    async def __aenter__(self) -> "SentenceHumanizer":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._rewriter.aclose()

    @property
    def model_id(self) -> str:
        return self._model_id

    async def humanize_selected(
        self,
        *,
        sentences: list[str],
        selected_indices: list[int],
        language_hint: str = "English",
        task_id: str = "",
    ) -> RewriteResult:
        if not selected_indices:
            return RewriteResult({}, self._model_id)

        system_prompt = build_humanize_system_prompt(language_hint)
        user_prompt = build_humanize_user_prompt(sentences, selected_indices)
        prompt = f"{system_prompt}\n\n{user_prompt}"

        raw_text, input_tokens, output_tokens = await self._rewriter._call_api_with_retry(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        )

        rewrites, diagnostics = self._rewriter._parse_response(
            raw_text,
            selected_indices,
            task_id=task_id,
        )
        parse_ok = (
            not diagnostics["missing_indices"]
            and not diagnostics["invalid_indices"]
            and not diagnostics["extra_indices"]
        )

        self._api_logger.log(
            task_id=task_id,
            model=self._model_id,
            prompt=prompt,
            raw_response=raw_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            parse_ok=parse_ok,
            error="" if parse_ok else diagnostics["error"],
        )

        return RewriteResult(
            rewrites,
            self._model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            prompt=prompt,
            missing_indices=diagnostics["missing_indices"],
            invalid_indices=diagnostics["invalid_indices"],
            extra_indices=diagnostics["extra_indices"],
            error=diagnostics["error"],
        )
