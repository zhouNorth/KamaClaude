import asyncio
import json
import os

import httpx2
from dotenv import load_dotenv

load_dotenv()


async def main() -> None:
    base = os.environ["ANTHROPIC_BASE_URL"].rstrip("/")
    headers = {
        "x-api-key": os.environ["ANTHROPIC_API_KEY"],
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    custom_raw = os.environ.get("ANTHROPIC_CUSTOM_HEADERS", "").strip()
    if custom_raw:
        try:
            headers.update(json.loads(custom_raw))
        except (json.JSONDecodeError, TypeError):
            pass

    body = {
        "model": os.environ.get("KAMA_LLM_DEFAULT_MODEL", "DeepSeek-V4-Flash"),
        "max_tokens": 64,
        "stream": True,
        "messages": [{"role": "user", "content": "say hi"}],
    }
    async with httpx2.AsyncClient() as client:
        async with client.stream(
            "POST", f"{base}/v1/messages", headers=headers, json=body
        ) as response:
            print("status:", response.status_code)
            print("content-type:", response.headers.get("content-type"))
            async for chunk in response.aiter_bytes():
                print(chunk.decode("utf-8", errors="replace"))


asyncio.run(main())
