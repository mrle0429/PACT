"""
LLM 改写统一入口。

这个文件保持很薄，只负责向外暴露稳定 API。
实际实现拆分在 `src/rewriters/` 中：
- `base.py`：公共基类、Prompt、解析、日志
- `factory.py`：provider 注册表与工厂
- 其余模块：各 provider 实现
"""
from __future__ import annotations

from .rewriters import (
    ApiCallLogger,
    BaseLLMRewriter,
    PROVIDER_REWRITER_CLASSES,
    RewriteResult,
    build_rewrite_prompt,
    create_rewriter,
)

__all__ = [
    "ApiCallLogger",
    "BaseLLMRewriter",
    "PROVIDER_REWRITER_CLASSES",
    "RewriteResult",
    "build_rewrite_prompt",
    "create_rewriter",
]
