from __future__ import annotations

import asyncio
import signal
from unittest.mock import Mock

from kama_claude.core.app import _install_signal_handlers


async def test_install_signal_handlers_falls_back_on_windows(
    monkeypatch,
) -> None:
    shutdown = asyncio.Event()
    loop = Mock()
    loop.add_signal_handler.side_effect = NotImplementedError
    loop.call_soon_threadsafe.side_effect = lambda callback: callback()
    installed: dict[int, object] = {}

    monkeypatch.setattr(
        signal,
        "signal",
        lambda signum, handler: installed.__setitem__(signum, handler),
    )

    _install_signal_handlers(loop, shutdown)

    assert signal.SIGINT in installed
    assert signal.SIGTERM in installed

    installed[signal.SIGINT](signal.SIGINT, None)  # type: ignore[operator]

    assert shutdown.is_set()
