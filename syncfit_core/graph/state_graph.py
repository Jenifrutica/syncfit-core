"""Directed State Graph of physiological states.

Models the transitions of the ovarian cycle (menstrual -> follicular ->
ovulatory -> luteal) and the gestational progression (trimester 1 -> 2 -> 3),
supporting weighted probabilistic edges to represent irregularities.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..enums import InferredPhase, Modality

CYCLE_STATES: tuple[InferredPhase, ...] = (
    InferredPhase.MENSTRUAL,
    InferredPhase.FOLLICULAR,
    InferredPhase.OVULATORY,
    InferredPhase.LUTEAL,
)

TRIMESTER_STATES: tuple[InferredPhase, ...] = (
    InferredPhase.TRIMESTER_1,
    InferredPhase.TRIMESTER_2,
    InferredPhase.TRIMESTER_3,
)


@dataclass(frozen=True)
class StateNode:
    id: str
    state: InferredPhase
    kind: str
    label: str


@dataclass(frozen=True)
class StateTransition:
    from_id: str
    to_id: str
    weight: float
    condition: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError("transition weight must be within [0, 1]")


@dataclass
class DirectedStateGraph:
    """A weighted directed graph over physiological states."""

    nodes: dict[str, StateNode] = field(default_factory=dict)
    edges: list[StateTransition] = field(default_factory=list)

    def add_node(self, node: StateNode) -> None:
        self.nodes[node.id] = node

    def add_edge(
        self,
        from_id: str,
        to_id: str,
        weight: float = 1.0,
        condition: str | None = None,
    ) -> None:
        if from_id not in self.nodes or to_id not in self.nodes:
            raise KeyError("both endpoints must exist before adding an edge")
        self.edges.append(
            StateTransition(from_id=from_id, to_id=to_id, weight=weight, condition=condition)
        )

    def successors(self, node_id: str) -> list[StateTransition]:
        return [edge for edge in self.edges if edge.from_id == node_id]

    def transition_probability(self, from_id: str, to_id: str) -> float:
        for edge in self.edges:
            if edge.from_id == from_id and edge.to_id == to_id:
                return edge.weight
        return 0.0

    def most_likely_next(self, node_id: str) -> StateNode | None:
        candidates = self.successors(node_id)
        if not candidates:
            return None
        best = max(candidates, key=lambda edge: edge.weight)
        return self.nodes[best.to_id]

    def node_for_state(self, state: InferredPhase) -> StateNode | None:
        for node in self.nodes.values():
            if node.state == state:
                return node
        return None


def default_cycle_graph() -> DirectedStateGraph:
    """Build the canonical cycle + gestation graph with weighted edges."""
    graph = DirectedStateGraph()
    for state in CYCLE_STATES:
        node_id = state.value.lower()
        graph.add_node(StateNode(node_id, state, "CYCLE_PHASE", state.value.title()))
    for state in TRIMESTER_STATES:
        node_id = state.value.lower()
        graph.add_node(StateNode(node_id, state, "TRIMESTER", state.value.replace("_", " ").title()))

    graph.add_edge("menstrual", "follicular", 0.95)
    graph.add_edge("follicular", "ovulatory", 0.90)
    graph.add_edge("ovulatory", "luteal", 0.88)
    graph.add_edge("luteal", "menstrual", 0.85, condition="no pregnancy detected")

    graph.add_edge("trimester_1", "trimester_2", 0.99)
    graph.add_edge("trimester_2", "trimester_3", 0.99)
    return graph


def infer_phase_from_day(
    modality: Modality | str, day_or_week: int, cycle_length_days: int = 28
) -> InferredPhase:
    """Deterministically map a day/week to a phase.

    Menstrual cycle: a 28-day reference cycle split into menstrual (1-5),
    follicular (6-13), ovulatory (14-16) and luteal (17+). Gestation: weeks 1-13
    are the first trimester, 14-27 the second and 28+ the third.

    For other cycle lengths the luteal phase keeps its ~14 days, so ovulation
    moves with the length (day L-14 to L-12) and the follicular phase absorbs
    the difference. The default of 28 keeps the reference split.
    """
    modality_value = modality if isinstance(modality, Modality) else Modality(modality)
    if day_or_week < 1:
        raise ValueError("day_or_week must be positive")
    if not 21 <= cycle_length_days <= 45:
        raise ValueError("cycle_length_days must be between 21 and 45")

    if modality_value is Modality.GESTATIONAL:
        if day_or_week <= 13:
            return InferredPhase.TRIMESTER_1
        if day_or_week <= 27:
            return InferredPhase.TRIMESTER_2
        return InferredPhase.TRIMESTER_3

    ovulation_start = cycle_length_days - 14
    if day_or_week <= 5:
        return InferredPhase.MENSTRUAL
    if day_or_week < ovulation_start:
        return InferredPhase.FOLLICULAR
    if day_or_week <= ovulation_start + 2:
        return InferredPhase.OVULATORY
    return InferredPhase.LUTEAL


__all__ = [
    "CYCLE_STATES",
    "TRIMESTER_STATES",
    "StateNode",
    "StateTransition",
    "DirectedStateGraph",
    "default_cycle_graph",
    "infer_phase_from_day",
]
