"""
配置管理模块 — 所有可调超参数集中在此，支持 .env 注入 API Key。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# LLM Provider 配置
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """单个 LLM 模型的调用参数。"""
    provider: Literal["openai", "anthropic", "gemini", "minimax", "dashscope"]
    model_id: str
    temperature: float = 0.7
    max_output_tokens: int = 2048
    requests_per_minute: int = 60   # 速率限制（RPM）


SUPPORTED_MODELS: dict[str, ModelConfig] = {

    "gemini-3.1-flash-lite-preview": ModelConfig(
        provider="gemini",
        model_id="gemini-3.1-flash-lite-preview",
        temperature=0.7,
        requests_per_minute=120,
    ),
    # --- MiniMax（Anthropic API 兼容）---
    "MiniMax-M2.5": ModelConfig(
        provider="minimax",
        model_id="MiniMax-M2.5",
        temperature=0.7,
        requests_per_minute=60,
    ),
    # --- DashScope / Qwen（OpenAI API 兼容）---
    "qwen3.5-plus": ModelConfig(
        provider="dashscope",
        model_id="qwen3.5-plus",
        temperature=0.7,
        requests_per_minute=60,
    ),
    "qwen3.5-flash": ModelConfig(
        provider="dashscope",
        model_id="qwen3.5-flash-2026-02-23",
        temperature=0.7,
        requests_per_minute=60,
    ),
}


# ---------------------------------------------------------------------------
# 数据集生成配置
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    """数据集构建的核心超参数。"""

    # --- 数据源（预采样 JSONL）---
    source_path: str = "data/human_texts_10k.jsonl"
    source_tag: str = "human_10k"

    # --- AI 浓度档位 ---
    # 0.0 = 纯人类基线, 1.0 = 全AI改写
    ai_ratios: list[float] = field(
        default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    )

    # --- 混合模式 ---
    mixing_modes: list[str] = field(
        default_factory=lambda: ["block_replace", "random_scatter"]
    )
    # 文档级模式池：每篇文档仅使用一种模式；在数据集内尽量均衡分配（两种模式时约 1:1）

    # --- LLM 改写 ---
    rewrite_model: str = "qwen3.5-plus"   # 使用的模型 key
    max_retries: int = 3                 # API 调用失败最大重试次数
    retry_wait_seconds: float = 5.0
    concurrent_requests: int = 8         # 并发 API 请求数

    # --- 标签计算 ---
    # block_replace 用 LIR（Token 长度占比）
    # random_scatter 同时计算 Jaccard 距离和余弦距离
    tokenizer_for_lir: str = "cl100k_base"   # tiktoken 编码器名
    ngram_n: int = 2                          # 余弦距离的 n-gram 大小

    # --- 输出 ---
    output_dir: str = "output"
    output_filename: str = "mixed_dataset.jsonl"
    checkpoint_dir: str = "output/checkpoints"

    # --- 复现性 ---
    random_seed: int = 42

    # ---------------------------------------------------------------------------
    # API Key（从环境变量读取，不硬编码）
    # ---------------------------------------------------------------------------
    @property
    def openai_api_key(self) -> str:
        return os.getenv("OPENAI_API_KEY", "")

    @property
    def anthropic_api_key(self) -> str:
        return os.getenv("ANTHROPIC_API_KEY", "")

    @property
    def gemini_api_key(self) -> str:
        return os.getenv("GEMINI_API_KEY", "")

    @property
    def minimax_api_key(self) -> str:
        return os.getenv("MINIMAX_API_KEY", "")

    @property
    def dashscope_api_key(self) -> str:
        return os.getenv("DASHSCOPE_API_KEY", "")

    def get_model_config(self) -> ModelConfig:
        if self.rewrite_model not in SUPPORTED_MODELS:
            raise ValueError(
                f"未知模型 '{self.rewrite_model}'，"
                f"可选: {list(SUPPORTED_MODELS.keys())}"
            )
        return SUPPORTED_MODELS[self.rewrite_model]
