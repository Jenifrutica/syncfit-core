"""Fenwick Tree (Binary Indexed Tree).

Optional, lighter alternative to a full segment tree for cumulative range
aggregates over the cycle/gestation timeline. Point updates and prefix queries
run in O(log n).
"""

from __future__ import annotations

from typing import Iterable


class FenwickTree:
    """1-indexed Fenwick Tree over numeric values.

    Supports point updates and prefix/range sums in O(log n).
    """

    __slots__ = ("_tree", "_n")

    def __init__(self, values: Iterable[float] | None = None, size: int = 0) -> None:
        if values is not None:
            data = list(values)
            self._n = len(data)
            self._tree = [0.0] * (self._n + 1)
            for index, value in enumerate(data, start=1):
                self._tree[index] += value
                parent = index + (index & -index)
                if parent <= self._n:
                    self._tree[parent] += self._tree[index]
        else:
            if size < 0:
                raise ValueError("size must be non-negative")
            self._n = size
            self._tree = [0.0] * (size + 1)

    @property
    def size(self) -> int:
        return self._n

    def add(self, index: int, delta: float) -> None:
        """Add `delta` to the 0-indexed position `index` in O(log n)."""
        if not 0 <= index < self._n:
            raise IndexError("index out of range")
        i = index + 1
        while i <= self._n:
            self._tree[i] += delta
            i += i & -i

    def prefix_sum(self, end: int) -> float:
        """Sum of positions [0, end] (inclusive). Use end=-1 for 0."""
        if end < 0:
            return 0.0
        if end >= self._n:
            raise IndexError("end out of range")
        i = end + 1
        total = 0.0
        while i > 0:
            total += self._tree[i]
            i -= i & -i
        return total

    def range_sum(self, start: int, end: int) -> float:
        """Sum of positions [start, end] (inclusive) in O(log n)."""
        if start > end:
            return 0.0
        return self.prefix_sum(end) - self.prefix_sum(start - 1)

    def range_mean(self, start: int, end: int) -> float:
        """Mean of positions [start, end] (inclusive)."""
        if start > end:
            raise ValueError("start must not be greater than end")
        count = end - start + 1
        return self.range_sum(start, end) / count


__all__ = ["FenwickTree"]
