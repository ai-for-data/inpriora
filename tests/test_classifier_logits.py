import json

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from sklearn.exceptions import NotFittedError

from inpriora import InprioraClassifier
from inpriora.core.config import ModelConfig
from inpriora.core.model import InprioraModel


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("classifier")
    config = ModelConfig(
        model_type="classifier",
        native_classes=10,
        embed_dim=8,
        column_blocks=1,
        column_heads=2,
        column_inducing_points=4,
        row_blocks=1,
        row_heads=2,
        row_cls_tokens=2,
        icl_blocks=1,
        icl_heads=2,
        ff_factor=2.0,
    )
    torch.manual_seed(42)
    model = InprioraModel(config)
    (path / "config.json").write_text(json.dumps(config.to_dict()))
    save_file(model.state_dict(), str(path / "model.safetensors"))
    return str(path)


@pytest.mark.parametrize("cache", [False, True, "repr"])
@pytest.mark.parametrize("n_classes", [2, 3, 10])
def test_logits_follow_ensemble_and_probability_semantics(checkpoint, cache, n_classes):
    rng = np.random.default_rng(7)
    X = rng.normal(size=(n_classes * 4, 3))
    y = np.array([f"class_{i:02d}" for i in range(n_classes)] * 4)
    model = InprioraClassifier(
        model_path=checkpoint,
        n_estimators=3,
        cache=cache,
        temperature=0.7,
        device="cpu",
        n_jobs=1,
    ).fit(X, y)
    sample = X[:5]
    proba_before = model.predict_proba(sample)
    raw = model.predict_logits(sample, return_raw=True)
    mean = model.predict_logits(sample)

    assert raw.shape == (3, len(sample), n_classes)
    assert mean.shape == (len(sample), n_classes)
    np.testing.assert_allclose(mean, raw.mean(axis=0), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(model._softmax(mean, model.temperature), proba_before, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(model.predict_proba(sample), proba_before)
    np.testing.assert_array_equal(model.predict(sample), model.classes_[proba_before.argmax(axis=1)])

    model.average_logits = False
    expected = np.mean([model._softmax(member, model.temperature) for member in raw], axis=0)
    np.testing.assert_allclose(model.predict_proba(sample), expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(model.predict_logits(sample), mean, rtol=0, atol=0)
    model.temperature = 2.3
    np.testing.assert_allclose(model.predict_logits(sample, return_raw=True), raw, rtol=0, atol=0)


def test_member_logits_are_aligned_to_class_order(checkpoint, monkeypatch):
    X = np.arange(36, dtype=float).reshape(12, 3)
    model = InprioraClassifier(model_path=checkpoint, n_estimators=2, cache=False, n_jobs=1)
    model.fit(X, np.tile(["a", "b", "c"], 4))
    outputs = np.array([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]], dtype=np.float32)
    patterns = [np.array([2, 0, 1]), np.array([1, 2, 0])]
    monkeypatch.setattr(model, "_predict_member_outputs", lambda X: (outputs, patterns))

    expected = np.array([[[3.0, 1.0, 2.0]], [[5.0, 6.0, 4.0]]])
    np.testing.assert_array_equal(model.predict_logits(X[:1], return_raw=True), expected)
    np.testing.assert_array_equal(model.predict_logits(X[:1]), expected.mean(axis=0))


def test_logits_require_fit_and_reject_hierarchical_classes(checkpoint, monkeypatch):
    model = InprioraClassifier(model_path=checkpoint, n_estimators=1, cache=False, n_jobs=1)
    with pytest.raises(NotFittedError):
        model.predict_logits(np.zeros((1, 3)))

    X = np.arange(66, dtype=float).reshape(22, 3)
    model.fit(X, np.tile(np.arange(11), 2))
    monkeypatch.setattr(model, "_predict_member_outputs", lambda X: pytest.fail("inference should not run"))
    with pytest.raises(NotImplementedError, match="hierarchical classification"):
        model.predict_logits(X[:1])
    with pytest.raises(NotImplementedError, match="hierarchical classification"):
        model.predict_logits(X[:1], return_raw=True)
