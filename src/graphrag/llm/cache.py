"""Tiny on-disk JSON cache, so re-ingesting a document never pays for the same LLM call twice."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class JsonCache:
    def __init__(self, directory: str | Path) -> None:
        self.dir = Path(directory)
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("LLM cache disabled, cannot create %s: %s", self.dir, exc)
            self.dir = None  # type: ignore[assignment]

    @staticmethod
    def key(*parts: str) -> str:
        return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()

    def _path(self, key: str) -> Path:
        return self.dir / key[:2] / f"{key}.json"

    def get(self, key: str) -> Any | None:
        if self.dir is None:
            return None
        path = self._path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def set(self, key: str, value: Any) -> None:
        if self.dir is None:
            return
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)  # atomic: a crash never leaves half-written JSON
