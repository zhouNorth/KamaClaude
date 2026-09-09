import asyncio
import os

import anthropic
from dotenv import load_dotenv

load_dotenv()


async def main() -> None:
    client = anthropic.AsyncAnthropic()
    response = await client.messages.create(
        model=os.environ.get("KAMA_LLM_DEFAULT_MODEL", "DeepSeek-V4-Flash"),
        max_tokens=64,
        messages=[{"role": "user", "content": "say hi"}],
    )
    print(
        "stop_reason:",
        response.stop_reason,
        "blocks:",
        [block.type for block in response.content],
        "usage:",
        response.usage.input_tokens,
        response.usage.output_tokens,
    )


asyncio.run(main())
