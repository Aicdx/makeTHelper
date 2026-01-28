from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import datetime

from src.signals.signals import SignalEvent


_logger = logging.getLogger(__name__)


_last_emit_key: str | None = None
_last_emit_at: datetime | None = None


def emit(event: SignalEvent) -> None:
    """Emit alert.

    De-duplicate identical alerts that happen very close together (e.g. both logger + print,
    or repeated emission in the same minute).
    """

    global _last_emit_key, _last_emit_at

    key = f"{event.symbol}|{event.ts:%Y-%m-%d %H:%M}|{event.signal}|{event.reason}"
    now = datetime.now()

    # If the exact same message repeats within 3 seconds, drop it.
    if _last_emit_key == key and _last_emit_at is not None and (now - _last_emit_at).total_seconds() < 3:
        return

    _last_emit_key = key
    _last_emit_at = now

    msg = (
        f"[{event.ts:%Y-%m-%d %H:%M}] {event.symbol} {event.signal}: {event.reason} "
        f"| close={event.close:.3f} dif={event.dif:.4f} dea={event.dea:.4f} hist={event.hist:.4f} "
        f"vol={event.vol if event.vol is not None else 'NA'} vma={event.vma if event.vma is not None else 'NA'}"
    )

    # Only print once; log is enough for OpenHands to capture.
    _logger.warning(msg)

