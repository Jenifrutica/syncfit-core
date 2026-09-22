"""Fixed-capacity circular buffer with O(1) insertion.

The technical document requires a Ring Buffer for the continuous 100 Hz PPG
stream in fixed static memory, avoiding RAM overload and heap fragmentation.
"""

from __future__ import annotations

from typing import Generic, Iterable, Iterator, TypeVar

import numpy as np

T = TypeVar("T")


class RingBuffer(Generic[T]):
    """A fixed-size circular buffer.

    Insertion is O(1). Once full, writing overwrites the oldest element.
    """

    __slots__ = ("_buffer", "_capacity", "_head", "_size")

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self._capacity = capacity
        self._buffer: list[T | None] = [None] * capacity
        self._head = 0
        self._size = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def size(self) -> int:
        return self._size

    @property
    def is_full(self) -> bool:
        return self._size == self._capacity

    def __len__(self) -> int:
        return self._size

    def append(self, item: T) -> T | None:
        """Insert an item in O(1). Returns the overwritten item, if any."""
        overwritten = self._buffer[self._head] if self.is_full else None
        self._buffer[self._head] = item
        self._head = (self._head + 1) % self._capacity
        if self._size < self._capacity:
            self._size += 1
        return overwritten

    def extend(self, items: Iterable[T]) -> None:
        for item in items:
            self.append(item)

    def latest(self) -> T | None:
        """Return the most recently appended item, or None when empty."""
        if self._size == 0:
            return None
        return self._buffer[(self._head - 1) % self._capacity]

    def to_list(self) -> list[T]:
        """Return the items in chronological order."""
        if self._size < self._capacity:
            start = 0
        else:
            start = self._head
        return [
            self._buffer[(start + i) % self._capacity]  # type: ignore[misc]
            for i in range(self._size)
        ]

    def to_numpy(self, dtype: type = float) -> np.ndarray:
        """Return the items as a NumPy array in chronological order."""
        return np.asarray(self.to_list(), dtype=dtype)

    def clear(self) -> None:
        self._buffer = [None] * self._capacity
        self._head = 0
        self._size = 0

    def __iter__(self) -> Iterator[T]:
        return iter(self.to_list())


__all__ = ["RingBuffer"]
