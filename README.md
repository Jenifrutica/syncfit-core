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

## Suggested structure

```
syncfit-core/
├── syncfit_core/
│   ├── dsp/            # filters, R-R detection, RMSSD
│   ├── features/       # feature vector construction
│   ├── models/         # training pipeline, serialized models
│   ├── load/           # k_load calculation
│   └── graph/          # Directed State Graph
├── tests/
├── notebooks/          # exploratory analysis only
├── pyproject.toml
└── README.md
```

## Stack

Python 3.11+, SciPy, NumPy, scikit-learn, XGBoost.

## Related repositories

- [`syncfit-contracts`](../syncfit-contracts) — I/O schemas.
- [`syncfit-ai-reasoning`](../syncfit-ai-reasoning) — consumes fatigue probability and `k_load`.
- [`syncfit-backend`](../syncfit-backend) — imports this library.
- [`syncfit-simulator`](../syncfit-simulator) — feeds test signals.

All code, comments, documentation and commits in this repository are written in English.
