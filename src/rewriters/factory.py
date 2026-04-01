"""Rewriter 工厂与 provider 注册表。"""
from __future__ import annotations

from pathlib import Path

from ..config import DatasetConfig
from .anthropic_compatible import AnthropicRewriter, MiniMaxRewriter
from .base import ApiCallLogger, BaseLLMRewriter
from .gemini import GeminiRewriter
from .ollama import OllamaRewriter
from .openai_compatible import (
    DashScopeRewriter,
    DeepSeekRewriter,
    DoubaoRewriter,
    OpenAIRewriter,
    OpenRouterRewriter,
)

PROVIDER_REWRITER_CLASSES: dict[str, type[BaseLLMRewriter]] = {
    "openai": OpenAIRewriter,
    "anthropic": AnthropicRewriter,
    "gemini": GeminiRewriter,
    "minimax": MiniMaxRewriter,
    "dashscope": DashScopeRewriter,
    "deepseek": DeepSeekRewriter,
    "doubao": DoubaoRewriter,
    "ollama": OllamaRewriter,
    "openrouter": OpenRouterRewriter,
}


def create_rewriter(
    cfg: DatasetConfig,
    api_logger: ApiCallLogger | None = None,
) -> BaseLLMRewriter:
    """
    根据 cfg.rewrite_model 自动实例化对应改写器。

    新增 provider 时，只需要：
    1. 新建一个 BaseLLMRewriter 子类
    2. 在 PROVIDER_REWRITER_CLASSES 中注册一行
    """
    model_cfg = cfg.get_model_config()

    if api_logger is None:
        log_dir = Path(cfg.output_dir) / "api_logs"
        run_name = f"{cfg.rewrite_model}_{cfg.source_tag}"
        api_logger = ApiCallLogger(log_dir, run_name)

    rewriter_cls = PROVIDER_REWRITER_CLASSES.get(model_cfg.provider)
    if rewriter_cls is None:
        raise ValueError(f"未知 provider: '{model_cfg.provider}'")
    return rewriter_cls(cfg, model_cfg, api_logger)
