from kama_claude.core.llm.base import LLMProvider
from kama_claude.core.llm.openai_compatible import OpenAICompatibleProvider
from kama_claude.core.llm.provider import AnthropicProvider
from kama_claude.core.llm.types import LlmResponse, ToolCallBlock, UsageStats

__all__ = [
    "AnthropicProvider", "LLMProvider", "LlmResponse", "ToolCallBlock", "UsageStats",
    "create_provider",
]


# 按配置创建聊天和会话压缩共用的模型客户端
def create_provider(provider: str, model: str) -> LLMProvider:
    if provider == "openai":
        return OpenAICompatibleProvider(model)
    if provider == "anthropic":
        return AnthropicProvider(model)
    raise SystemExit(f"Unsupported KAMA_LLM_PROVIDER: {provider}")
