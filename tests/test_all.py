import pytest
import numpy as np
from afi import Perception, Distortion, Freedom, GlobalState

def test_canonical_perception():
    assert Perception.canonical(2, 1.0) == 1.0
    assert Perception.canonical(4, 2.0) == 4.0
    assert Perception.canonical(-1, 5.0) == 1.0

def test_canonical_distortion():
    D_val = Distortion.canonical(2.0, 2.0, 1.0, 1.0, 1.0)
    assert D_val == 4.0

def test_f_scalar():
    assert Freedom.f1_scalar(10.0, 2.0) == 5.0
    assert Freedom.f1_scalar(1.0, 0.0) >= 10000.0

def test_harmonic_aggregation():
    f_domains = np.array([1.0, 0.5, 0.2])
    weights = np.array([1.0, 1.0, 1.0])
    f_global = GlobalState.f_global_harmonic(f_domains, weights)
    assert np.isclose(f_global, 0.375, atol=0.01)

def test_regime_classifier():
    assert GlobalState.regime_classifier(1.0, 1.0) == "PASSIVE"
    assert GlobalState.regime_classifier(2.0, 1.0) == "ACTIVE"
    assert GlobalState.regime_classifier(2.0, 2.0) == "INTELLIGENT"
