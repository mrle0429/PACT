"""
配置管理模块 — 所有可调超参数集中在此，支持 .env 注入 API Key。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# LLM Provider 配置
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """单个 LLM 模型的调用参数。"""
    provider: Literal["openai", "anthropic", "gemini", "minimax", "dashscope", "deepseek", "doubao", "ollama", "openrouter"]
    model_id: str
    temperature: float = 0.7
    max_output_tokens: int = 2048
    requests_per_minute: int = 60   # 速率限制（RPM）


SUPPORTED_MODELS: dict[str, ModelConfig] = {
    # --- Ollama（本地 HTTP API）---
    "llama4-fast:latest": ModelConfig(
        provider="ollama",
        model_id="llama4-fast:latest",
        temperature=0.2,
        requests_per_minute=30,
    ),
    "gemma4": ModelConfig(
        provider="ollama",
        model_id="gemma4",
        temperature=0.7,
        requests_per_minute=30,
    ),

    "gemini-3.1-flash-lite-preview": ModelConfig(
        provider="gemini",
        model_id="gemini-3.1-flash-lite-preview",
        temperature=0.7,
        requests_per_minute=120,
    ),
    # --- MiniMax（Anthropic API 兼容）---
    "MiniMax-M2.7": ModelConfig(
        provider="minimax",
        model_id="MiniMax-M2.7",
        temperature=0.7,
        requests_per_minute=60,
    ),
    # --- DashScope / Qwen（OpenAI API 兼容）---
    "qwen3.5-plus": ModelConfig(
        provider="dashscope",
        model_id="qwen3.5-plus",
        temperature=0.1,
        requests_per_minute=60,
    ),
    "qwen3.5-flash": ModelConfig(
        provider="dashscope",
        model_id="qwen3.5-flash-2026-02-23",
        temperature=0.7,
        requests_per_minute=60,
    ),
    # --- OpenRouter / Qwen 3.6（OpenAI API 兼容）---
    "qwen3.6-plus-preview-free": ModelConfig(
        provider="openrouter",
        model_id="qwen/qwen3.6-plus-preview:free",
        temperature=0.1,
        requests_per_minute=60,
    ),
    # --- OpenRouter / Claude Haiku 4.5（OpenAI API 兼容）---
    "claude-haiku-4.5": ModelConfig(
        provider="openrouter",
        model_id="anthropic/claude-haiku-4.5",
        temperature=0.1,
        requests_per_minute=60,
    ),
    # --- DeepSeek（OpenAI API 兼容）---
    "DeepSeek-V3.2": ModelConfig(
        provider="deepseek",
        model_id="deepseek-chat",
        temperature=0.7
    ),
    # --- 豆包 / Doubao（火山方舟 Ark，OpenAI API 兼容）---
    "doubao-seed-2-0-pro": ModelConfig(
        provider="doubao",
        model_id="doubao-seed-2-0-pro-260215",
        temperature=0.7,
        requests_per_minute=60,
    ),
}


@dataclass
class DatasetConfig:
    """
    数据集构建配置。
    """

    # --- 运行时可变参数 ---
    rewrite_model: str
    source_path: str = "data/human_texts_1k.cleaned.jsonl"
    output_dir: str = "output"
    concurrent_requests: int = 8
    random_seed: int = 42

    # --- 由运行时参数派生 ---
    source_tag: str = field(init=False)
    checkpoint_dir: str = field(init=False)

    # --- 当前工作流固定约定 ---
    ai_ratios: list[float] = field(default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], init=False)
    mixing_modes: list[str] = field(default_factory=lambda: ["block_replace", "random_scatter"], init=False)
    max_retries: int = field(default=3, init=False)
    retry_wait_seconds: float = field(default=5.0, init=False)
    tokenizer_for_lir: str = field(default="cl100k_base", init=False)
    ngram_n: int = field(default=2, init=False)
    output_filename: str = field(default="mixed_dataset.jsonl", init=False)

    def __post_init__(self) -> None:
        self.source_tag = Path(self.source_path).stem
        self.checkpoint_dir = str(Path(self.output_dir) / "checkpoints")

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

    @property
    def deepseek_api_key(self) -> str:
        return os.getenv("DEEPSEEK_API_KEY", "")

    @property
    def ark_api_key(self) -> str:
        return os.getenv("ARK_API_KEY", "")

    @property
    def ollama_base_url(self) -> str:
        return os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/api")

    @property
    def ollama_keep_alive(self) -> str:
        return os.getenv("OLLAMA_KEEP_ALIVE", "5m")

    @property
    def openrouter_api_key(self) -> str:
        return os.getenv("OPENROUTER_API_KEY", "")

    @property
    def openrouter_base_url(self) -> str:
        return os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    def get_model_config(self) -> ModelConfig:
        if self.rewrite_model not in SUPPORTED_MODELS:
            raise ValueError(
                f"未知模型 '{self.rewrite_model}'，"
                f"可选: {list(SUPPORTED_MODELS.keys())}"
            )
        return SUPPORTED_MODELS[self.rewrite_model]
