import numpy as np

from src.data import (
    FAITHFUL_PATH,
    FAITHFUL_SHA256,
    _sha256,
    load_faithful,
    make_synthetic,
    standardize,
)


def test_synthetic_is_reproducible():
    X1, z1 = make_synthetic(seed=42)
    X2, z2 = make_synthetic(seed=42)
    assert X1.shape == (300, 2)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(z1, z2)


def test_faithful_shape_and_checksum():
    X = load_faithful()
    assert X.shape == (272, 2)
    assert not np.isnan(X).any()
    assert _sha256(FAITHFUL_PATH) == FAITHFUL_SHA256


def test_standardize():
    X_std, _, _ = standardize(load_faithful())
    np.testing.assert_allclose(X_std.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(X_std.std(axis=0), 1)
