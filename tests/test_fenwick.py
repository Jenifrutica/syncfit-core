import pytest

from syncfit_core.structures import FenwickTree


def test_build_from_values_and_prefix():
    tree = FenwickTree([1, 2, 3, 4, 5])
    assert tree.size == 5
    assert tree.prefix_sum(0) == 1
    assert tree.prefix_sum(4) == 15
    assert tree.prefix_sum(-1) == 0.0


def test_range_sum_and_mean():
    tree = FenwickTree([1, 2, 3, 4, 5])
    assert tree.range_sum(1, 3) == 9
    assert tree.range_mean(1, 3) == 3.0
    assert tree.range_sum(3, 1) == 0.0


def test_point_update():
    tree = FenwickTree(size=4)
    tree.add(0, 10)
    tree.add(3, 5)
    assert tree.range_sum(0, 3) == 15
    assert tree.range_sum(1, 2) == 0


def test_bounds():
    tree = FenwickTree([1, 2, 3])
    with pytest.raises(IndexError):
        tree.add(3, 1)
    with pytest.raises(IndexError):
        tree.prefix_sum(5)
    with pytest.raises(ValueError):
        tree.range_mean(2, 1)
