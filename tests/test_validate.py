"""Tests for shared validation helpers."""

import numpy as np
import pytest

from derivlab._validate import (
    broadcast_float_arrays,
    scalar_or_array,
    validate_kind,
    validate_non_negative,
    validate_positive,
    validate_sigma,
    validate_style,
    validate_T,
)


def test_validate_kind_and_style():
    assert validate_kind("Call") == "call"
    with pytest.raises(ValueError, match="kind must be"):
        validate_kind("swap")
    assert validate_style("European") == "european"
    with pytest.raises(ValueError, match="style must be"):
        validate_style("bermudan")


def test_validate_numeric_helpers():
    validate_positive("x", 1.0)
    with pytest.raises(ValueError, match="x must be > 0"):
        validate_positive("x", 0.0)
    validate_non_negative("x", 0.0)
    validate_sigma(0.1)
    validate_T(0.5)
    validate_T(0.0, allow_zero=True)


def test_broadcast_and_scalar_or_array():
    target, s, k, t = broadcast_float_arrays([1.0, 2.0], 100.0, [90.0, 95.0], 0.5)
    assert target.shape == (2,)
    assert s.shape == (2,)
    assert scalar_or_array(np.array(1.5)) == 1.5
    arr = np.array([1.0, 2.0])
    assert np.array_equal(scalar_or_array(arr), arr)
