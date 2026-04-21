"""
标签精确计算模块。

三个互补指标对所有混合模式统一计算：
  - LIR (Token 占比)      — AI 句子 token 数 / 全文 token 数，直接度量 AI 浓度
  - Jaccard Distance       — 原文 vs 混合文本的词汇集合差异，度量词汇替换程度
  - sentence-level Jaccard Distance — 仅对 AI 句子计算的平均词汇集合差异，度量 AI 句子质量
  - Cosine Distance (n-gram) — 原文 vs 混合文本的 n-gram TF 向量距离，度量风格偏移
"""
from __future__ import annotations

import math
import re
from collections import Counter

from .config import DatasetConfig
from .sentence_processor import split_into_sentences


# ---------------------------------------------------------------------------
# 文本预处理（标点清洗）
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def _normalize_words(text: str) -> list[str]:
    """小写 + 去标点 + 空格切分，返回词列表。"""
    cleaned = _PUNCT_RE.sub(" ", text.lower())
    return cleaned.split()


# ---------------------------------------------------------------------------
# Token 计数
# ---------------------------------------------------------------------------

_encoder_cache: dict[str, object] = {}


def _get_token_encoder(encoding_name: str):
    """懒加载 tiktoken 编码器（带缓存），失败时直接抛异常。"""
    if encoding_name in _encoder_cache:
        encoder = _encoder_cache[encoding_name]
        if encoder is None:
            raise RuntimeError(f"tiktoken 编码器不可用: {encoding_name}")
        return encoder
    try:
        import tiktoken
        enc = tiktoken.get_encoding(encoding_name)
        _encoder_cache[encoding_name] = enc
        return enc
    except Exception as exc:
        _encoder_cache[encoding_name] = None
        raise RuntimeError(
            f"无法加载 tiktoken 编码器 '{encoding_name}'，"
            "LIR token 统计要求精确计数，不能回退。"
        ) from exc


def count_tokens(text: str, encoder) -> int:
    """统计文本 token 数。"""
    return len(encoder.encode(text))


# ---------------------------------------------------------------------------
# 指标一：LIR（Large-model Involvement Ratio）
# ---------------------------------------------------------------------------

def compute_lir(
    ai_indices: list[int],
    mixed_sentences: list[str],
    encoding_name: str = "cl100k_base",
) -> float:
    """
    LIR = T_LLM / T_total

    T_LLM:   混合文档中 AI 改写句子的 token 数之和
    T_total: 混合文档全文 token 总数

    适用于所有混合模式，是最通用的 AI 浓度度量。

    Args:
        ai_indices:     已被 AI 改写的句子下标（0-indexed）
        mixed_sentences: 回填后的最终句子列表
        encoding_name:  tiktoken 编码器名称

    Returns:
        LIR 浮点值 ∈ [0.0, 1.0]
    """
    if not mixed_sentences:
        return 0.0

    encoder = _get_token_encoder(encoding_name)

    # 全文 token 总数
    full_text = " ".join(mixed_sentences)
    t_total = count_tokens(full_text, encoder)
    if t_total == 0:
        return 0.0

    # AI 句子 token 数（校验下标越界）
    valid_indices = [i for i in ai_indices if 0 <= i < len(mixed_sentences)]
    t_llm = sum(count_tokens(mixed_sentences[i], encoder) for i in valid_indices)

    lir = t_llm / t_total
    return round(min(lir, 1.0), 6)


# ---------------------------------------------------------------------------
# 指标二：Jaccard Distance（词汇集合差异）
# ---------------------------------------------------------------------------

def _word_set(text: str) -> set[str]:
    """经过标点清洗的词集合。"""
    return set(_normalize_words(text))


def compute_jaccard_distance(original_text: str, mixed_text: str) -> float:
    """
    Jaccard Distance = 1 - |A ∩ B| / |A ∪ B|

    比较人类原文与混合文本在词汇集合上的差异。
    返回值 ∈ [0.0, 1.0]，值越大词汇替换越多。
    """
    a = _word_set(original_text)
    b = _word_set(mixed_text)
    union = a | b
    if not union:
        return 0.0
    intersection = a & b
    jaccard_sim = len(intersection) / len(union)
    return round(1.0 - jaccard_sim, 6)


# ---------------------------------------------------------------------------
# 指标三：sentence-level Jaccard Distance（仅 AI 句子）
# ---------------------------------------------------------------------------
def compute_sentence_jaccard(
    original_text: str,
    mixed_text: str,
    sentence_labels: list[int],
) -> float | None:
    """
    仅对 sentence_labels == 1 的句子计算平均 Jaccard Distance。

    若 original / mixed / labels 三者句子数无法安全对齐，则返回 None。
    """
    ai_indices = [idx for idx, label in enumerate(sentence_labels) if int(label) == 1]
    if not ai_indices:
        return 0.0

    original_sentences = split_into_sentences(original_text)
    mixed_sentences = split_into_sentences(mixed_text)

    if len(original_sentences) != len(mixed_sentences) or len(original_sentences) != len(sentence_labels):
        return None

    scores = [
        compute_jaccard_distance(original_sentences[idx], mixed_sentences[idx])
        for idx in ai_indices
    ]
    if not scores:
        return 0.0
    return round(sum(scores) / len(scores), 6)


# ---------------------------------------------------------------------------
# 指标四：余弦距离（基于 n-gram TF 向量）
# ---------------------------------------------------------------------------

def _ngram_tf(text: str, n: int) -> Counter:
    """提取经过标点清洗的 n-gram 词频向量。"""
    words = _normalize_words(text)
    if len(words) < n:
        return Counter()
    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return Counter(ngrams)


def compute_cosine_distance(
    original_text: str,
    mixed_text: str,
    n: int = 2,
) -> float:
    """
    基于 n-gram TF 向量计算余弦距离 = 1 - 余弦相似度。
    返回值 ∈ [0.0, 1.0]，值越大风格偏移越大。
    """
    vec_a = _ngram_tf(original_text, n)
    vec_b = _ngram_tf(mixed_text, n)

    # 并集 key
    all_keys = set(vec_a) | set(vec_b)
    if not all_keys:
        return 0.0

    dot = sum(vec_a[k] * vec_b[k] for k in all_keys)
    norm_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    norm_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))

    if norm_a == 0 or norm_b == 0:
        return 1.0  # 一方为零向量，视为完全不同

    cosine_sim = dot / (norm_a * norm_b)
    return round(1.0 - min(cosine_sim, 1.0), 6)


# ---------------------------------------------------------------------------
# 统一标签计算入口
# ---------------------------------------------------------------------------

def compute_labels(
    original_text: str,
    mixed_sentences: list[str],
    ai_indices: list[int],
    cfg: DatasetConfig,
) -> dict:
    """
    对所有混合模式统一计算全部三个指标。

    Returns:
        {
            "lir":                float,  # AI token 占比
            "jaccard_distance":   float,  # 词汇集合差异
            "cosine_distance":    float,  # n-gram 风格偏移
            "sentence_jaccard":   float | None,  # AI 句级平均 Jaccard Distance
        }
    """
    mixed_text = " ".join(mixed_sentences)
    sentence_labels = [1 if idx in set(ai_indices) else 0 for idx in range(len(mixed_sentences))]

    lir_val = compute_lir(ai_indices, mixed_sentences, cfg.tokenizer_for_lir)
    jac_val = compute_jaccard_distance(original_text, mixed_text)
    cos_val = compute_cosine_distance(original_text, mixed_text, cfg.ngram_n)
    sent_jac_val = compute_sentence_jaccard(original_text, mixed_text, sentence_labels)

    return {
        "lir": lir_val,
        "jaccard_distance": jac_val,
        "cosine_distance": cos_val,
        "sentence_jaccard": sent_jac_val,
    }
