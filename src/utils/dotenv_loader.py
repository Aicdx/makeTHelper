from __future__ import annotations

import os
from pathlib import Path


def load_env() -> None:
    """Best-effort load env vars from local files.

    Some environments block writing dotfiles like .env, so we support a non-dot
    fallback file: ./src/env.local

    Load order (first found wins; override=False):
    - ./.env
    - ./src/.env
    - ./src/env.local
    """

    if os.getenv("DOTENV_DISABLE") == "1":
        return

    try:
        from dotenv import load_dotenv
    except Exception:
        return

    cwd = Path.cwd()
    candidates = [cwd / ".env", cwd / "src" / ".env", cwd / "src" / "env.local"]

    for p in candidates:
        try:
            if p.exists():
                load_dotenv(dotenv_path=str(p), override=False)
        except Exception:
            continue
