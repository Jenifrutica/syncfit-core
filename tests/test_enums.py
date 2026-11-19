"""The core enums stay wire-compatible with syncfit-contracts."""

import json
from enum import Enum
from pathlib import Path

import pytest

from syncfit_core.enums import CyclePhase, FatigueLevel, InferredPhase, Modality, Trimester

ENUMS: list[type[Enum]] = [Modality, CyclePhase, Trimester, InferredPhase, FatigueLevel]
# Sibling checkout used in local development; CI only has this repo.
COMMON_SCHEMA = Path(__file__).resolve().parents[2] / "syncfit-contracts" / "schemas" / "common.schema.json"


@pytest.mark.parametrize("enum", ENUMS)
def test_values_equal_names_and_serialize_as_strings(enum):
    for member in enum:
        assert member.value == member.name
        assert json.dumps(member) == f'"{member.value}"'


def test_inferred_phase_is_cycle_phases_plus_trimesters():
    expected = [p.value for p in CyclePhase] + [t.value for t in Trimester]
    assert [p.value for p in InferredPhase] == expected


@pytest.mark.skipif(not COMMON_SCHEMA.exists(), reason="syncfit-contracts is not checked out next to this repo")
@pytest.mark.parametrize("enum", ENUMS)
def test_values_match_the_contracts_schema(enum):
    defs = json.loads(COMMON_SCHEMA.read_text())["$defs"]
    assert [m.value for m in enum] == defs[enum.__name__]["enum"]
