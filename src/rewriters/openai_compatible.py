"""OpenAI 协议族 provider 实现。"""
from __future__ import annotations

from typing import Any

from .base import ApiCallLogger, BaseLLMRewriter
from ..config import DatasetConfig, ModelConfig
from ..utils import get_logger

logger = get_logger(__name__)


def _build_openai_messages(user_prompt: str, system_prompt: str | None = None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages


class OpenAIRewriter(BaseLLMRewriter):
    """使用 openai 官方异步客户端。"""

    def __init__(
        self,
        cfg: DatasetConfig,
        model_cfg: ModelConfig,
        api_logger: ApiCallLogger | None = None,
    ):
        super().__init__(cfg, model_cfg, api_logger)
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError("请安装 openai: pip install openai") from exc
        if not cfg.openai_api_key:
            raise ValueError("未设置 OPENAI_API_KEY 环境变量。")
        self._client = AsyncOpenAI(api_key=cfg.openai_api_key)

    async def _call_api(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
    ) -> tuple[str, int, int]:
        response = await self._client.chat.completions.create(
            model=self.model_cfg.model_id,
            messages=_build_openai_messages(user_prompt, system_prompt),
            temperature=self.model_cfg.temperature,
            max_tokens=self.model_cfg.max_output_tokens,
        )
        content = response.choices[0].message.content or ""
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        return content, input_tokens, output_tokens


class DashScopeRewriter(BaseLLMRewriter):
    """使用 OpenAI SDK 调用阿里云百炼 DashScope 模型。"""

    DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    _JSON_SCHEMA_RESPONSE_FORMAT: dict[str, Any] = {
        "type": "json_schema",
        "json_schema": {
            "name": "rewrites_by_sentence_index",
            "description": "Map from 1-indexed sentence number to rewritten sentence text.",
            "schema": {
                "type": "object",
                "propertyNames": {"pattern": "^[1-9][0-9]*$"},
                "additionalProperties": {"type": "string", "minLength": 1},
            },
            "strict": True,
        },
    }
    _JSON_OBJECT_RESPONSE_FORMAT: dict[str, str] = {"type": "json_object"}

    def __init__(
        self,
        cfg: DatasetConfig,
        model_cfg: ModelConfig,
        api_logger: ApiCallLogger | None = None,
    ):
        super().__init__(cfg, model_cfg, api_logger)
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError("DashScope 使用 OpenAI SDK，请安装: pip install openai") from exc
        if not cfg.dashscope_api_key:
            raise ValueError("未设置 DASHSCOPE_API_KEY 环境变量。")
        self._client = AsyncOpenAI(
            api_key=cfg.dashscope_api_key,
            base_url=self.DASHSCOPE_BASE_URL,
        )

    async def _call_api(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
    ) -> tuple[str, int, int]:
        try:
            response = await self._client.chat.completions.create(
                model=self.model_cfg.model_id,
                messages=_build_openai_messages(user_prompt, system_prompt),
                extra_body={"enable_thinking": False},
                temperature=self.model_cfg.temperature,
                max_tokens=self.model_cfg.max_output_tokens,
                response_format=self._JSON_SCHEMA_RESPONSE_FORMAT,
            )
        except Exception as exc:
            err = str(exc).lower()
            if ("response_format" in err) or ("json_schema" in err) or ("unsupported" in err):
                logger.warning("DashScope json_schema response_format 不可用，回退到 json_object。")
                response = await self._client.chat.completions.create(
                    model=self.model_cfg.model_id,
                    messages=_build_openai_messages(user_prompt, system_prompt),
                    temperature=self.model_cfg.temperature,
                    max_tokens=self.model_cfg.max_output_tokens,
                    response_format=self._JSON_OBJECT_RESPONSE_FORMAT,
                )
            else:
                raise
        content = response.choices[0].message.content or ""
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        return content, input_tokens, output_tokens


class DeepSeekRewriter(BaseLLMRewriter):
    """使用 OpenAI SDK 调用 DeepSeek 模型。"""

    DEEPSEEK_BASE_URL = "https://api.deepseek.com"
    _JSON_OBJECT_RESPONSE_FORMAT: dict[str, str] = {"type": "json_object"}

    def __init__(
        self,
        cfg: DatasetConfig,
        model_cfg: ModelConfig,
        api_logger: ApiCallLogger | None = None,
    ):
        super().__init__(cfg, model_cfg, api_logger)
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError("DeepSeek 使用 OpenAI SDK，请安装: pip install openai") from exc
        if not cfg.deepseek_api_key:
            raise ValueError("未设置 DEEPSEEK_API_KEY 环境变量。")
        self._client = AsyncOpenAI(
            api_key=cfg.deepseek_api_key,
            base_url=self.DEEPSEEK_BASE_URL,
        )

    async def _call_api(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
    ) -> tuple[str, int, int]:
        response = await self._client.chat.completions.create(
            model=self.model_cfg.model_id,
            messages=_build_openai_messages(user_prompt, system_prompt),
            temperature=self.model_cfg.temperature,
            max_tokens=self.model_cfg.max_output_tokens,
            response_format=self._JSON_OBJECT_RESPONSE_FORMAT,
        )
        content = response.choices[0].message.content or ""
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        return content, input_tokens, output_tokens


class DoubaoRewriter(BaseLLMRewriter):
    """使用 OpenAI SDK 调用火山方舟（Volcengine Ark）豆包模型。"""

    ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
    _THINKING_DISABLED_EXTRA_BODY: dict[str, Any] = {
        "thinking": {"type": "disabled"},
    }

    def __init__(
        self,
        cfg: DatasetConfig,
        model_cfg: ModelConfig,
        api_logger: ApiCallLogger | None = None,
    ):
        super().__init__(cfg, model_cfg, api_logger)
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError("Doubao 使用 OpenAI SDK，请安装: pip install openai") from exc
        if not cfg.ark_api_key:
            raise ValueError("未设置 ARK_API_KEY 环境变量。")
        self._client = AsyncOpenAI(
            api_key=cfg.ark_api_key,
            base_url=self.ARK_BASE_URL,
        )

    async def _call_api(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
    ) -> tuple[str, int, int]:
        response = await self._client.chat.completions.create(
            model=self.model_cfg.model_id,
            messages=_build_openai_messages(user_prompt, system_prompt),
            temperature=self.model_cfg.temperature,
            max_tokens=self.model_cfg.max_output_tokens,
            extra_body=self._THINKING_DISABLED_EXTRA_BODY,
        )
        content = response.choices[0].message.content or ""
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        return content, input_tokens, output_tokens


class OpenRouterRewriter(BaseLLMRewriter):
    """使用 OpenAI SDK 调用 OpenRouter 上的模型。"""

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    _JSON_OBJECT_RESPONSE_FORMAT: dict[str, str] = {"type": "json_object"}
    _REASONING_ENABLED_EXTRA_BODY: dict[str, Any] = {
        "reasoning": {
            "enabled": True,
            "exclude": True,
        },
    }

    def __init__(
        self,
        cfg: DatasetConfig,
        model_cfg: ModelConfig,
        api_logger: ApiCallLogger | None = None,
    ):
        super().__init__(cfg, model_cfg, api_logger)
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError("OpenRouter 使用 OpenAI SDK，请安装: pip install openai") from exc
        if not cfg.openrouter_api_key:
            raise ValueError("未设置 OPENROUTER_API_KEY 环境变量。")
        self._client = AsyncOpenAI(
            api_key=cfg.openrouter_api_key,
            base_url=cfg.openrouter_base_url or self.OPENROUTER_BASE_URL,
        )

    async def _call_api(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
    ) -> tuple[str, int, int]:
        try:
            response = await self._client.chat.completions.create(
                model=self.model_cfg.model_id,
                messages=_build_openai_messages(user_prompt, system_prompt),
                temperature=self.model_cfg.temperature,
                max_tokens=self.model_cfg.max_output_tokens,
                response_format=self._JSON_OBJECT_RESPONSE_FORMAT,
                extra_body=self._REASONING_ENABLED_EXTRA_BODY,
            )
        except Exception as exc:
            err = str(exc).lower()
            if ("response_format" in err) or ("json_object" in err) or ("unsupported" in err):
                logger.warning("OpenRouter json_object response_format 不可用，回退到普通 chat 调用。")
                response = await self._client.chat.completions.create(
                    model=self.model_cfg.model_id,
                    messages=_build_openai_messages(user_prompt, system_prompt),
                    temperature=self.model_cfg.temperature,
                    max_tokens=self.model_cfg.max_output_tokens,
                    extra_body=self._REASONING_ENABLED_EXTRA_BODY,
                )
            else:
                raise
        content = response.choices[0].message.content or ""
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        return content, input_tokens, output_tokens
