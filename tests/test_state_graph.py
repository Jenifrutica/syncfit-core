import pytest

from syncfit_core.enums import InferredPhase, Modality
from syncfit_core.graph import default_cycle_graph, infer_phase_from_day


def test_default_graph_structure():
    graph = default_cycle_graph()
    assert len(graph.nodes) == 7
    assert graph.transition_probability("ovulatory", "luteal") == 0.88
    assert graph.transition_probability("menstrual", "ovulatory") == 0.0


def test_most_likely_next():
    graph = default_cycle_graph()
    nxt = graph.most_likely_next("follicular")
    assert nxt is not None
    assert nxt.state is InferredPhase.OVULATORY
    assert graph.most_likely_next("trimester_3") is None


def test_node_for_state():
    graph = default_cycle_graph()
    node = graph.node_for_state(InferredPhase.TRIMESTER_2)
    assert node is not None and node.kind == "TRIMESTER"


@pytest.mark.parametrize(
    "day,expected",
    [
        (2, InferredPhase.MENSTRUAL),
        (10, InferredPhase.FOLLICULAR),
        (15, InferredPhase.OVULATORY),
        (22, InferredPhase.LUTEAL),
        (40, InferredPhase.LUTEAL),
    ],
)
def test_infer_menstrual_phase(day, expected):
    assert infer_phase_from_day(Modality.MENSTRUAL_CYCLE, day) is expected


@pytest.mark.parametrize(
    "week,expected",
    [
        (5, InferredPhase.TRIMESTER_1),
        (18, InferredPhase.TRIMESTER_2),
        (30, InferredPhase.TRIMESTER_3),
    ],
)
def test_infer_gestational_phase(week, expected):
    assert infer_phase_from_day("GESTATIONAL", week) is expected


def test_infer_invalid_day():
    with pytest.raises(ValueError):
        infer_phase_from_day(Modality.MENSTRUAL_CYCLE, 0)


@pytest.mark.parametrize(
    "length,day,expected",
    [
        (28, 13, InferredPhase.FOLLICULAR),
        (28, 14, InferredPhase.OVULATORY),
        (28, 17, InferredPhase.LUTEAL),
        (35, 16, InferredPhase.FOLLICULAR),
        (35, 21, InferredPhase.OVULATORY),
        (35, 24, InferredPhase.LUTEAL),
        (24, 10, InferredPhase.OVULATORY),
        (24, 3, InferredPhase.MENSTRUAL),
    ],
)
def test_infer_phase_with_cycle_length(length, day, expected):
    assert infer_phase_from_day(Modality.MENSTRUAL_CYCLE, day, cycle_length_days=length) is expected


@pytest.mark.parametrize("length", [20, 46])
def test_infer_phase_invalid_cycle_length(length):
    with pytest.raises(ValueError):
        infer_phase_from_day(Modality.MENSTRUAL_CYCLE, 5, cycle_length_days=length)
