"""Ollama provider 实现。"""
from __future__ import annotations

from .base import ApiCallLogger, BaseLLMRewriter
from ..config import DatasetConfig, ModelConfig


class OllamaRewriter(BaseLLMRewriter):
    """
    使用 Ollama 原生 HTTP API 调用本地/隧道映射模型。
    """

    def __init__(
        self,
        cfg: DatasetConfig,
        model_cfg: ModelConfig,
        api_logger: ApiCallLogger | None = None,
    ):
        super().__init__(cfg, model_cfg, api_logger)
        try:
            import httpx
        except ImportError as exc:
            raise ImportError("Ollama 接入需要 httpx，请安装: pip install httpx") from exc

        self._client = httpx.AsyncClient(
            base_url=cfg.ollama_base_url.rstrip("/"),
            timeout=300.0,
            trust_env=False,
        )

    async def _call_api(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
    ) -> tuple[str, int, int]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        payload = {
            "model": self.model_cfg.model_id,
            "messages": messages,
            "options": {
                "temperature": self.model_cfg.temperature,
                "num_predict": self.model_cfg.max_output_tokens,
            },
            "format": "json",
            "keep_alive": self.cfg.ollama_keep_alive,
            "stream": False,
        }
        if self.model_cfg.model_id.startswith("gemma4"):
            payload["think"] = False

        response = await self._client.post("/chat", json=payload)
        response.raise_for_status()

        payload = response.json()
        if payload.get("error"):
            raise RuntimeError(str(payload["error"]))

        message = payload.get("message") or {}
        content = message.get("content", "") or ""
        input_tokens = int(payload.get("prompt_eval_count") or 0)
        output_tokens = int(payload.get("eval_count") or 0)
        return content, input_tokens, output_tokens

    async def aclose(self) -> None:
        await self._client.aclose()
