from __future__ import annotations

import hashlib
from typing import Any, Iterable, Protocol

from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict


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


class MessageIdArchive:
    """For view mode: the full history stays in graph state, so a message's id is its key."""

    def __init__(self, messages: Iterable[BaseMessage] = ()) -> None:
        self._by_id = {m.id: m for m in messages if m.id}

    def put(self, message: BaseMessage) -> str:
        if not message.id:
            raise ValueError("MessageIdArchive needs messages with ids")
        self._by_id[message.id] = message
        return message.id

    def get(self, key: str) -> BaseMessage:
        return self._by_id[key]


class StoreArchive:
    """Persists originals in a LangGraph `BaseStore`, so they survive restarts and threads."""

    def __init__(self, store: Any, namespace: tuple[str, ...] = ("ctxsieve", "archive")) -> None:
        self._store = store
        self._namespace = namespace

    def put(self, message: BaseMessage) -> str:
        key = archive_key(message)
        self._store.put(self._namespace, key, {"message": message_to_dict(message)})
        return key

    def get(self, key: str) -> BaseMessage:
        item = self._store.get(self._namespace, key)
        if item is None:
            raise KeyError(key)
        return messages_from_dict([item.value["message"]])[0]
