import pytest

from syncfit_core.structures import SlidingWindow


def test_window_eviction_and_order():
    window: SlidingWindow[int] = SlidingWindow(maxlen=3)
    window.extend([1, 2, 3, 4])
    assert window.to_list() == [2, 3, 4]
    assert window.latest() == 4
    assert window.is_full()


def test_mean_and_numpy():
    window = SlidingWindow(maxlen=4, items=[1.0, 2.0, 3.0])
    assert window.mean() == pytest.approx(2.0)
    assert window.to_numpy().shape == (3,)
    assert len(window) == 3


def test_clear_and_empty_mean():
    window: SlidingWindow[float] = SlidingWindow(maxlen=2)
    assert window.mean() == 0.0
    window.append(1.0)
    window.clear()
    assert window.latest() is None


def test_invalid_maxlen():
    with pytest.raises(ValueError):
        SlidingWindow(maxlen=0)
