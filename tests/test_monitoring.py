import numpy as np
import pandas as pd
import pytest

from src.monitoring.check_drift import calculate_psi


def test_identical_distributions_have_zero_psi():
    reference = pd.Series(np.arange(1000, dtype=float))

    psi = calculate_psi(reference, reference.copy())

    assert psi == pytest.approx(0.0, abs=1e-12)


def test_shifted_numeric_distribution_has_high_psi():
    reference = pd.Series(np.arange(1000, dtype=float))
    current = reference + 2000

    psi = calculate_psi(reference, current)

    assert np.isfinite(psi)
    assert psi > 0.25


def test_new_category_has_finite_high_psi():
    reference = pd.Series(["A"] * 500 + ["B"] * 500)
    current = pd.Series(["C"] * 1000)

    psi = calculate_psi(reference, current, categorical=True)

    assert np.isfinite(psi)
    assert psi > 0.25
