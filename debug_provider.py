from dotenv import load_dotenv

load_dotenv()

import asyncio

from kama_claude.core.config import get_config
from kama_claude.core.events.bus import EventBus
from kama_claude.core.llm.provider import AnthropicProvider


async def main() -> None:
    provider = AnthropicProvider(get_config().llm.default_model)
    response = await provider.chat(
        [{"role": "user", "content": "say hi"}],
        [],
        EventBus(),
        "diag-001",
    )
    print("OK", response.text, response.stop_reason)


asyncio.run(main())
