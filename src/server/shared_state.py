from __future__ import annotations

from typing import Optional

from src.data.bar_store import BarStore
from src.server.data_bus import DataBus

# Global instances to be shared between the main loop and the server
data_bus = DataBus()
bar_store: Optional[BarStore] = None
