from __future__ import annotations

import hashlib
from typing import Protocol

from langchain_core.messages import BaseMessage


class Archive(Protocol):
    def put(self, message: BaseMessage) -> str: ...

    def get(self, key: str) -> BaseMessage: ...


def archive_key(message: BaseMessage) -> str:
    """Content-derived key, so archiving the same message twice yields the same key."""
    payload = message.model_dump_json().encode()
    return hashlib.sha256(payload).hexdigest()[:16]


class InMemoryArchive:
    def __init__(self) -> None:
        self._items: dict[str, BaseMessage] = {}

    def put(self, message: BaseMessage) -> str:
        key = archive_key(message)
        self._items[key] = message
        return key

    def get(self, key: str) -> BaseMessage:
        return self._items[key]

    def __len__(self) -> int:
        return len(self._items)
