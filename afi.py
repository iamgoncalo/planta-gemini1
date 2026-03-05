import numpy as np
from typing import Dict

class Perception:
    @staticmethod
    def canonical(N: float, T: float, P_floor: float = 1.0) -> float:
        if N <= 0 or T <= 0: return P_floor
        return max(P_floor, np.log2(max(2.0, N)) * T)

class Distortion:
    @staticmethod
    def canonical(R: float, O: float, Tb: float, C: float, M: float, 
                  exponents: Dict[str, float] = None, D_floor: float = 1.0) -> float:
        ex = exponents or {'alpha': 1.0, 'beta': 1.0, 'gamma': 1.0, 'delta': 1.0, 'epsilon': 1.0}
        d_val = (max(1.0, R)**ex['alpha'] * max(1.0, O)**ex['beta'] * max(1.0, Tb)**ex['gamma'] * max(1.0, C)**ex['delta'] * max(1.0, M)**ex['epsilon'])
        return max(D_floor, d_val)

class Freedom:
    F_MIN = 0.01
    @staticmethod
    def f1_scalar(P: float, D: float) -> float:
        return max(Freedom.F_MIN, P / max(1e-5, D))

class GlobalState:
    @staticmethod
    def f_global_harmonic(f_domains: np.ndarray, weights: np.ndarray) -> float:
        f_domains = np.maximum(Freedom.F_MIN, f_domains)
        weights = weights / np.sum(weights)
        return max(Freedom.F_MIN, 1.0 / np.sum(weights / f_domains))
