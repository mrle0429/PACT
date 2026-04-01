"""Gemini provider 实现。"""
from __future__ import annotations

from .base import ApiCallLogger, BaseLLMRewriter
from ..config import DatasetConfig, ModelConfig


class GeminiRewriter(BaseLLMRewriter):
    """使用 google-genai 库（新版 SDK，原生 async）。"""

    def __init__(
        self,
        cfg: DatasetConfig,
        model_cfg: ModelConfig,
        api_logger: ApiCallLogger | None = None,
    ):
        super().__init__(cfg, model_cfg, api_logger)
        try:
            from google import genai

            self._genai = genai
        except ImportError as exc:
            raise ImportError("请安装 google-genai: pip install google-genai") from exc
        if not cfg.gemini_api_key:
            raise ValueError("未设置 GEMINI_API_KEY 环境变量。")
        self._client = genai.Client(api_key=cfg.gemini_api_key)

    async def _call_api(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
    ) -> tuple[str, int, int]:
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=self.model_cfg.temperature,
            max_output_tokens=self.model_cfg.max_output_tokens,
            system_instruction=system_prompt,
        )
        response = await self._client.aio.models.generate_content(
            model=self.model_cfg.model_id,
            contents=user_prompt,
            config=config,
        )
        content = response.text or ""
        usage = getattr(response, "usage_metadata", None)
        input_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
        output_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0
        return content, input_tokens, output_tokens
