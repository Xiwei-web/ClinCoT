from __future__ import annotations

from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Simple name -> callable/class registry."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._store: dict[str, T] = {}

    def register(self, key: str | None = None):
        def _decorator(obj: T) -> T:
            k = key or getattr(obj, "__name__", None)
            if not k:
                raise ValueError("Registry key is empty")
            if k in self._store:
                raise KeyError(f"{k} already registered in {self.name}")
            self._store[k] = obj
            return obj

        return _decorator

    def add(self, key: str, obj: T) -> None:
        if key in self._store:
            raise KeyError(f"{key} already registered in {self.name}")
        self._store[key] = obj

    def get(self, key: str) -> T:
        if key not in self._store:
            raise KeyError(f"{key} not found in {self.name}")
        return self._store[key]

    def has(self, key: str) -> bool:
        return key in self._store

    def keys(self) -> list[str]:
        return sorted(self._store.keys())

    def build(self, key: str, *args, **kwargs):
        obj = self.get(key)
        if callable(obj):
            return obj(*args, **kwargs)
        raise TypeError(f"Registered object for {key} is not callable")

    def __contains__(self, key: str) -> bool:
        return self.has(key)

    def __len__(self) -> int:
        return len(self._store)
