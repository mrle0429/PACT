"""
句子处理模块：分句、句子索引选择、混合文本回填。

本模块只负责“文本到句子”这一层的纯本地处理：
- 分句
- 选择哪些句子要被改写
- 将改写结果安全回填到原句列表

"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

import pysbd


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

MixingMode = Literal["block_replace", "random_scatter"]


@dataclass
class SentenceSelection:
    """
    描述一次句子选取结果。

    Attributes:
        sentences:       原始句子列表（未修改）
        selected_indices: 被选中进行 AI 改写的句子下标（0-indexed）
        mode:            混合模式
        target_ratio:    目标 AI 浓度档位（0.0 ~ 1.0）
    """
    sentences: list[str]
    selected_indices: list[int]
    mode: MixingMode
    target_ratio: float

    @property
    def n(self) -> int:
        return len(self.sentences)

    @property
    def k(self) -> int:
        return len(self.selected_indices)

    def build_mixed_sentences(self, rewrites: dict[int, str]) -> list[str]:
        """
        将 API 返回的改写句子（{原索引: 改写文本}）安全回填。
        未选中 / 未被改写的句子保持原样。
        """
        result = list(self.sentences)
        for idx, new_sentence in rewrites.items():
            if 0 <= idx < self.n and new_sentence and new_sentence.strip():
                result[idx] = new_sentence.strip()
        return result

    def build_mixed_text(self, rewrites: dict[int, str]) -> str:
        """
        将 API 返回的改写句子（{原索引: 改写文本}）安全回填，
        拼接为最终混合文档。未选中 / 未被改写的句子保持原样。
        """
        return " ".join(self.build_mixed_sentences(rewrites))

    def sentence_label_array(self, rewrites: dict[int, str]) -> list[int]:
        """
        返回句子级 0/1 标签数组。
        - 1 = 本句已被 AI 改写（且改写非空）
        - 0 = 保持原始人类文本
        """
        labels = [0] * self.n
        for idx in self.selected_indices:
            if idx in rewrites and rewrites[idx].strip():
                labels[idx] = 1
        return labels


# ---------------------------------------------------------------------------
# 分句
# ---------------------------------------------------------------------------

_segmenter = pysbd.Segmenter(language="en", clean=False)


def split_into_sentences(text: str) -> list[str]:
    """
    使用 PySBD 将文本分割为句子列表，并过滤空句。
    """
    return [s.strip() for s in _segmenter.segment(text) if s.strip()]


# ---------------------------------------------------------------------------
# 索引选择策略
# ---------------------------------------------------------------------------

def select_block_indices(n: int, k: int, rng: random.Random) -> list[int]:
    """
    Block Replace（异质化混合）：随机选一个起始点，取连续 k 个句子。
    边界处理：起始点 ∈ [0, n - k]，保证不越界。
    """
    if k == 0:
        return []
    if k >= n:
        return list(range(n))
    start = rng.randint(0, n - k)
    return list(range(start, start + k))


def select_scatter_indices(n: int, k: int, rng: random.Random) -> list[int]:
    """
    Random Scatter（同质化混合）：无放回随机抽取 k 个不连续的句子索引。
    返回排序后的下标列表，便于后续处理。
    """
    if k == 0:
        return []
    if k >= n:
        return list(range(n))
    return sorted(rng.sample(range(n), k))


# ---------------------------------------------------------------------------
# 主接口
# ---------------------------------------------------------------------------

def create_sentence_selection(
    sentences: list[str],
    target_ratio: float,
    mode: MixingMode,
    rng: random.Random,
) -> SentenceSelection:
    """
    给定句子列表、目标 AI 浓度和混合模式，返回 SentenceSelection 对象。

    Args:
        sentences:     已分割的句子列表
        target_ratio:  目标 AI 浓度 ∈ [0.0, 1.0]
        mode:          "block_replace" 或 "random_scatter"
        rng:           可控随机数生成器（保证复现性）
    """
    n = len(sentences)
    k = round(target_ratio * n)
    k = max(0, min(k, n))   # clamp

    if mode == "block_replace":
        indices = select_block_indices(n, k, rng)
    else:
        indices = select_scatter_indices(n, k, rng)

    return SentenceSelection(
        sentences=sentences,
        selected_indices=indices,
        mode=mode,
        target_ratio=target_ratio,
    )


def enumerate_variants(
    text: str,
    ai_ratios: list[float],
    mixing_modes: list[str],
    rng: random.Random,
) -> list[SentenceSelection]:
    """
    为一篇文档生成所有 (ratio × mode) 组合的 SentenceSelection。

    特殊规则：
    - ratio = 0.0 → 纯人类文本基线，仅生成一次（无需区分模式）
    - ratio = 1.0 → 全 AI 改写，两种模式等价，仅生成一次
    """
    sentences = split_into_sentences(text)
    variants: list[SentenceSelection] = []

    for ratio in ai_ratios:
        modes_to_use: list[str]
        if ratio == 0.0 or ratio == 1.0:
            # 边界档位：两种模式等价，只生成一份（用第一个模式名占位）
            modes_to_use = [mixing_modes[0]] if mixing_modes else ["block_replace"]
        else:
            modes_to_use = list(mixing_modes)

        for mode in modes_to_use:
            sel = create_sentence_selection(sentences, ratio, mode, rng)  # type: ignore[arg-type]
            variants.append(sel)

    return variants
