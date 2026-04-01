"""Anthropic 协议族 provider 实现。"""
from __future__ import annotations

from .base import ApiCallLogger, BaseLLMRewriter
from ..config import DatasetConfig, ModelConfig


class AnthropicRewriter(BaseLLMRewriter):
    """使用 anthropic 官方异步客户端。"""

    def __init__(
        self,
        cfg: DatasetConfig,
        model_cfg: ModelConfig,
        api_logger: ApiCallLogger | None = None,
    ):
        super().__init__(cfg, model_cfg, api_logger)
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("请安装 anthropic: pip install anthropic") from exc
        if not cfg.anthropic_api_key:
            raise ValueError("未设置 ANTHROPIC_API_KEY 环境变量。")
        self._client = anthropic.AsyncAnthropic(api_key=cfg.anthropic_api_key)

    async def _call_api(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
    ) -> tuple[str, int, int]:
        response = await self._client.messages.create(
            model=self.model_cfg.model_id,
            max_tokens=self.model_cfg.max_output_tokens,
            temperature=self.model_cfg.temperature,
            system=system_prompt or "",
            messages=[{"role": "user", "content": user_prompt}],
        )
        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text
        input_tokens = response.usage.input_tokens if response.usage else 0
        output_tokens = response.usage.output_tokens if response.usage else 0
        return content, input_tokens, output_tokens


class MiniMaxRewriter(BaseLLMRewriter):
    """使用 Anthropic SDK 调用 MiniMax 模型。"""

    MINIMAX_BASE_URL = "https://api.minimaxi.com/anthropic"
    _THINKING_DISABLED: dict[str, str] = {"type": "disabled"}

    def __init__(
        self,
        cfg: DatasetConfig,
        model_cfg: ModelConfig,
        api_logger: ApiCallLogger | None = None,
    ):
        super().__init__(cfg, model_cfg, api_logger)
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("MiniMax 使用 Anthropic SDK，请安装: pip install anthropic") from exc
        if not cfg.minimax_api_key:
            raise ValueError("未设置 MINIMAX_API_KEY 环境变量。")
        self._client = anthropic.AsyncAnthropic(
            api_key=cfg.minimax_api_key,
            base_url=self.MINIMAX_BASE_URL,
        )

    async def _call_api(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
    ) -> tuple[str, int, int]:
        try:
            response = await self._client.messages.create(
                model=self.model_cfg.model_id,
                max_tokens=self.model_cfg.max_output_tokens,
                temperature=self.model_cfg.temperature,
                system=system_prompt or "",
                thinking=self._THINKING_DISABLED,
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": user_prompt}],
                    }
                ],
            )
        except Exception as exc:
            err = str(exc).lower()
            if "thinking" in err and ("disabled" in err or "unsupported" in err or "invalid" in err):
                response = await self._client.messages.create(
                    model=self.model_cfg.model_id,
                    max_tokens=self.model_cfg.max_output_tokens,
                    temperature=self.model_cfg.temperature,
                    system=system_prompt or "",
                    messages=[
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": user_prompt}],
                        }
                    ],
                )
            else:
                raise
        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text
        input_tokens = response.usage.input_tokens if response.usage else 0
        output_tokens = response.usage.output_tokens if response.usage else 0
        return content, input_tokens, output_tokens
