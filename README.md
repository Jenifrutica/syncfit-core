# SyncFit Core

Deterministic numerical engine of SyncFit Edge: digital signal processing, the trained tabular model, and the load multiplier. It is a pure Python library, imported by the backend, with no network or UI concerns.

This is the **trained, on-premise half** of the artificial intelligence. The cloud reasoning half lives in [`syncfit-ai-reasoning`](../syncfit-ai-reasoning).

## Purpose

Clean the raw PPG signal, compute autonomic biomarkers, and produce an objective, reproducible load adjustment. Because the calculation is deterministic, it never depends on a generative model.

## What belongs here

- **DSP** (`dsp/`): SciPy/NumPy Butterworth IIR band-pass filter (0.5 Hz – 5.0 Hz), R-R peak detection, RMSSD computation, artifact rejection.
- **Feature extraction** (`features/`): builds the numeric vector `[modality, day/week, ΔT, RMSSD, % isometric loss]`.
- **Trained model** (`models/`): supervised tabular classifier (Random Forest / XGBoost), training pipeline and serialized artifacts.
- **Load logic** (`load/`): central-fatigue probability and multiplier `k_load ∈ [0.70, 1.05]`.
- **Physiological state model** (`graph/`): the Directed State Graph.
- Unit tests with reproducible fixtures.

## What does NOT belong here

- API routes, WebSocket servers or any network service.
- Prompt engineering or LLM calls.
- Hardware drivers or UI rendering.

## Data Structures

| Structure | Complexity | Purpose |
|-----------|:----------:|---------|
| **Ring Buffer** | O(1) insert | Python-side buffer for the continuous 100 Hz PPG stream. |
| **Sliding-window deque** | O(1) append | Segmenting the signal for filtering and peak detection. |
| **Directed State Graph** | Graph traversal | Menstrual → follicular → ovulatory → luteal; trimester 1 → 2 → 3, with weighted probabilistic edges for irregularities. |
| **Fenwick Tree (BIT)** | O(log n) | Optional: lighter range aggregates when a full segment tree is unnecessary. |

The Ring Buffer is mandatory per the technical document, which specifies it in both C++ and Python.

## Repository layout

```
syncfit-core/
├── syncfit_core/
│   ├── enums.py             # Modality, phases, trimesters, fatigue levels
│   ├── structures/          # RingBuffer, SlidingWindow, FenwickTree
│   ├── dsp/                 # Butterworth filter, R-R detection, RMSSD
│   ├── features/            # feature-vector construction
│   ├── models/              # fatigue classifier + training pipeline
│   ├── load/                # deterministic k_load multiplier
│   ├── graph/               # Directed State Graph
│   ├── pipeline.py          # SyncFitEngine (end-to-end)
│   └── contracts_adapter.py # optional bridge to syncfit-contracts
├── tests/
├── notebooks/               # exploratory analysis only
├── pyproject.toml
└── README.md
```

## Usage

```bash
pip install -e ".[dev]"
pytest
```

```python
from syncfit_core import SyncFitEngine, train_default_model

model = train_default_model(n_samples=4000, seed=42)
engine = SyncFitEngine(model, window_size=256)

# Ingest a 100 Hz PPG window and evaluate biomarkers.
engine.ingest(ppg_samples)
result = engine.evaluate(
    modality="MENSTRUAL_CYCLE",
    day_or_week=14,
    delta_temperature_c=0.42,
    isometric_force_loss_pct=12.8,
)
print(result.as_dict())
# {'phase_inferred': 'OVULATORY', 'fatigue_level': ..., 'k_load_multiplier': 0.72, ...}
```

Train and persist a model from the command line:

```bash
syncfit-train --samples 4000 --backend RANDOM_FOREST --output models/artifacts/fatigue_model.joblib
```

> `syncfit-contracts` is optional. When installed
> (`pip install -e ".[contracts]"`, which pulls the package from the
> [`syncfit-contracts`](https://github.com/Jenifrutica/syncfit-contracts) repository),
> `syncfit_core.contracts_adapter` validates the adapted-routine payload against
> the shared schema.

## Data Structures

| Structure | Complexity | Purpose |
|-----------|:----------:|---------|
| **Ring Buffer** | O(1) insert | Python-side buffer for the continuous 100 Hz PPG stream. |
| **Sliding Window** (deque) | O(1) append | Segmenting the signal for filtering and peak detection. |
| **Directed State Graph** | Graph traversal | Menstrual → follicular → ovulatory → luteal; trimester 1 → 2 → 3, with weighted probabilistic edges for irregularities. |
| **Fenwick Tree (BIT)** | O(log n) | Optional: lighter range aggregates when a full segment tree is unnecessary. |

The Ring Buffer is mandatory per the technical document, which specifies it in both C++ and Python.

## Stack

Python 3.11+, NumPy, SciPy, scikit-learn, joblib; optional XGBoost and `syncfit-contracts`.

## Tasks

> **Language: Python 3.11+ (mandatory).**

### Requirements

- [x] Implement the Butterworth IIR band-pass filter (0.5 Hz – 5.0 Hz).
- [x] Implement R-R peak detection.
- [x] Implement the RMSSD computation.
- [x] Implement artifact rejection.
- [x] Build the feature vector `[modality, day/week, ΔT, RMSSD, % isometric loss]`.
- [x] Train the supervised tabular model (Random Forest / XGBoost).
- [x] Compute the central-fatigue probability.
- [x] Compute the load multiplier `k_load ∈ [0.70, 1.05]`.
- [x] Implement the **Directed State Graph** for phase and trimester states.
- [x] Implement the **Ring Buffer** and the **sliding-window deque**.
- [x] Serialize model artifacts.
- [x] Write unit tests with reproducible fixtures.

## Related repositories

- [`syncfit-contracts`](../syncfit-contracts) — I/O schemas.
- [`syncfit-ai-reasoning`](../syncfit-ai-reasoning) — consumes fatigue probability and `k_load`.
- [`syncfit-backend`](../syncfit-backend) — imports this library.
- [`syncfit-simulator`](../syncfit-simulator) — feeds test signals.

All code, comments, documentation and commits in this repository are written in English.
