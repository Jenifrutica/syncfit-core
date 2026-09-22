import numpy as np
import pytest

from syncfit_core.structures import RingBuffer


def test_capacity_and_size():
    rb: RingBuffer[int] = RingBuffer(capacity=3)
    assert rb.capacity == 3
    assert rb.size == 0
    assert not rb.is_full
    rb.extend([1, 2])
    assert rb.size == 2
    rb.append(3)
    assert rb.is_full
    assert len(rb) == 3


def test_overwrites_oldest_in_order():
    rb: RingBuffer[int] = RingBuffer(capacity=3)
    rb.extend([1, 2, 3])
    overwritten = rb.append(4)
    assert overwritten == 1
    assert rb.to_list() == [2, 3, 4]
    assert rb.latest() == 4


def test_to_numpy_and_iteration():
    rb: RingBuffer[float] = RingBuffer(capacity=4)
    rb.extend([1.0, 2.0, 3.0])
    assert np.allclose(rb.to_numpy(), [1.0, 2.0, 3.0])
    assert list(rb) == [1.0, 2.0, 3.0]


def test_clear_and_latest_empty():
    rb: RingBuffer[int] = RingBuffer(capacity=2)
    assert rb.latest() is None
    rb.extend([1, 2])
    rb.clear()
    assert rb.size == 0
    assert rb.latest() is None


def test_invalid_capacity():
    with pytest.raises(ValueError):
        RingBuffer(capacity=0)
