"""Sliding-window buffer backed by `collections.deque`.

Used to segment the filtered signal for windowed DSP (e.g. rolling RMSSD and
feature aggregation) with O(1) append and automatic eviction of old samples.
"""

from __future__ import annotations

from collections import deque
from typing import Generic, Iterable, Iterator, TypeVar

import numpy as np

T = TypeVar("T")


class SlidingWindow(Generic[T]):
    """A fixed-length window that evicts the oldest item on overflow."""

    __slots__ = ("_deque", "_maxlen")

    def __init__(self, maxlen: int, items: Iterable[T] | None = None) -> None:
        if maxlen <= 0:
            raise ValueError("maxlen must be a positive integer")
        self._maxlen = maxlen
        self._deque: deque[T] = deque(items or (), maxlen=maxlen)

    @property
    def maxlen(self) -> int:
        return self._maxlen

    def append(self, item: T) -> None:
        self._deque.append(item)

    def extend(self, items: Iterable[T]) -> None:
        self._deque.extend(items)

    def to_list(self) -> list[T]:
        return list(self._deque)

    def to_numpy(self, dtype: type = float) -> np.ndarray:
        return np.asarray(self._deque, dtype=dtype)

    def latest(self) -> T | None:
        return self._deque[-1] if self._deque else None

    def is_full(self) -> bool:
        return len(self._deque) == self._maxlen

    def mean(self) -> float:
        if not self._deque:
            return 0.0
        return float(np.mean(self.to_numpy()))

    def clear(self) -> None:
        self._deque.clear()

    def __len__(self) -> int:
        return len(self._deque)

    def __iter__(self) -> Iterator[T]:
        return iter(self._deque)


__all__ = ["SlidingWindow"]
