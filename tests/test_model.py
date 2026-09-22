import numpy as np

from syncfit_core.models import (
    FatigueModel,
    ModelBackend,
    make_synthetic_dataset,
    train_default_model,
)


def test_synthetic_dataset_shape_and_labels():
    features, labels = make_synthetic_dataset(n_samples=500, seed=7)
    assert features.shape == (500, 5)
    assert set(np.unique(labels)).issubset({0, 1})


def test_train_and_predict_probabilities():
    model = train_default_model(n_samples=2000, seed=3)
    assert model.is_fitted
    features, _ = make_synthetic_dataset(n_samples=10, seed=99)
    probabilities = model.predict_fatigue_probability(features)
    assert probabilities.shape == (10,)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_unfitted_model_raises():
    model = FatigueModel()
    try:
        model.predict_fatigue_probability(np.zeros((1, 5)))
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_save_and_load_roundtrip(tmp_path):
    model = train_default_model(n_samples=1000, seed=5)
    path = model.save(tmp_path / "model.joblib")
    loaded = FatigueModel.load(path)
    features, _ = make_synthetic_dataset(n_samples=5, seed=11)
    assert np.allclose(
        model.predict_fatigue_probability(features),
        loaded.predict_fatigue_probability(features),
    )


def test_reproducible_training():
    a = train_default_model(n_samples=800, seed=42)
    b = train_default_model(n_samples=800, seed=42)
    features, _ = make_synthetic_dataset(n_samples=20, seed=1)
    assert np.allclose(
        a.predict_fatigue_probability(features),
        b.predict_fatigue_probability(features),
    )


def test_backend_enum():
    assert ModelBackend.RANDOM_FOREST.value == "RANDOM_FOREST"
    assert ModelBackend.XGBOOST.value == "XGBOOST"
