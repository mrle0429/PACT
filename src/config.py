"""
配置管理模块
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# LLM Provider 配置
# ---------------------------------------------------------------------------

ProviderName = Literal[
    "openai",
    "anthropic",
    "gemini",
    "minimax",
    "dashscope",
    "deepseek",
    "doubao",
    "ollama",
    "openrouter",
]

DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_OUTPUT_TOKENS = 2048
DEFAULT_REQUESTS_PER_MINUTE = 60


@dataclass(frozen=True)
class ModelParameters:
    """LLM 调用参数。默认全局统一，必要时才按模型覆盖。"""

    temperature: float = DEFAULT_TEMPERATURE
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    requests_per_minute: int = DEFAULT_REQUESTS_PER_MINUTE   # 速率限制（RPM）

    def with_overrides(
        self,
        *,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        requests_per_minute: int | None = None,
    ) -> ModelParameters:
        return replace(
            self,
            temperature=self.temperature if temperature is None else temperature,
            max_output_tokens=(
                self.max_output_tokens
                if max_output_tokens is None
                else max_output_tokens
            ),
            requests_per_minute=(
                self.requests_per_minute
                if requests_per_minute is None
                else requests_per_minute
            ),
        )

    def validate(self) -> None:
        if not 0 <= self.temperature <= 2:
            raise ValueError(f"temperature 必须在 0 到 2 之间: {self.temperature}")
        if self.max_output_tokens <= 0:
            raise ValueError(f"max_output_tokens 必须大于 0: {self.max_output_tokens}")
        if self.requests_per_minute <= 0:
            raise ValueError(f"requests_per_minute 必须大于 0: {self.requests_per_minute}")


DEFAULT_MODEL_PARAMETERS = ModelParameters()


@dataclass(frozen=True)
class ModelConfig:
    """单个 LLM 模型的 provider 身份与默认调用参数。"""

    provider: ProviderName
    model_id: str
    params: ModelParameters = DEFAULT_MODEL_PARAMETERS

    @property
    def temperature(self) -> float:
        return self.params.temperature

    @property
    def max_output_tokens(self) -> int:
        return self.params.max_output_tokens

    @property
    def requests_per_minute(self) -> int:
        return self.params.requests_per_minute


def _model(
    provider: ProviderName,
    model_id: str,
    *,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
    requests_per_minute: int | None = None,
) -> ModelConfig:
    """
    创建模型配置。

    大多数模型直接继承 DEFAULT_MODEL_PARAMETERS；只有确实需要不同限流、
    温度或输出长度时，才在这里传入覆盖值。
    """
    return ModelConfig(
        provider=provider,
        model_id=model_id,
        params=DEFAULT_MODEL_PARAMETERS.with_overrides(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            requests_per_minute=requests_per_minute,
        ),
    )


SUPPORTED_MODELS: dict[str, ModelConfig] = {
    # --- OpenRouter ---
    "llama4-fast:latest": _model(
        provider="openrouter",
        model_id="meta-llama/llama-4-scout",
    ),

    # --- Gemini ---
    "gemma4": _model(
        provider="gemini",
        model_id="gemma-4-31b-it",
    ),

    "gemini-3.1-flash-lite-preview": _model(
        provider="gemini",
        model_id="gemini-3.1-flash-lite-preview",
    ),
    # --- MiniMax（Anthropic API 兼容）---
    "MiniMax-M2.7": _model(
        provider="minimax",
        model_id="MiniMax-M2.7",
    ),
    # --- DashScope / Qwen（OpenAI API 兼容）---
    "qwen3.5-plus": _model(
        provider="dashscope",
        model_id="qwen3.5-plus",
    ),
    "qwen3.6-plus": _model(
        provider="dashscope",
        model_id="qwen3.6-plus",
    ),
    "qwen3.5-flash": _model(
        provider="dashscope",
        model_id="qwen3.5-flash",
    ),
    # --- OpenRouter / Qwen 3.6（OpenAI API 兼容）---
    "qwen3.6-plus-preview-free": _model(
        provider="openrouter",
        model_id="qwen/qwen3.6-plus-preview:free",
    ),
    # --- OpenRouter / Claude Haiku 4.5（OpenAI API 兼容）---
    "claude-haiku-4.5": _model(
        provider="openrouter",
        model_id="anthropic/claude-haiku-4.5",
    ),
    "gpt-5.4": _model(
        provider="openrouter",
        model_id="openai/gpt-5.4",
    ),
    # --- DeepSeek（OpenAI API 兼容）---
    "DeepSeek-V3.2": _model(
        provider="deepseek",
        model_id="deepseek-chat",
    ),
    # --- 豆包 / Doubao（火山方舟 Ark，OpenAI API 兼容）---
    "doubao-seed-2-0-pro": _model(
        provider="doubao",
        model_id="doubao-seed-2-0-pro-260215",
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
    temperature: float | None = None
    max_output_tokens: int | None = None
    requests_per_minute: int | None = None

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
        model_cfg = SUPPORTED_MODELS[self.rewrite_model]
        params = model_cfg.params.with_overrides(
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            requests_per_minute=self.requests_per_minute,
        )
        params.validate()

        return replace(model_cfg, params=params)
