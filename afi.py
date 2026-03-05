import numpy as np
from typing import Dict

class Perception:
    @staticmethod
    def canonical(N: float, T: float, P_floor: float = 1.0) -> float:
        if N <= 0 or T <= 0: return P_floor
        return max(P_floor, np.log2(max(2.0, N)) * T)

    @staticmethod
    def alt1_linear(N: float, T: float, P_floor: float = 1.0) -> float:
        return max(P_floor, N * T)

    @staticmethod
    def alt2_log_additive(N: float, T: float, P_floor: float = 1.0) -> float:
        return max(P_floor, np.log(max(2.0, N + T)))

class Distortion:
    @staticmethod
    def canonical(R: float, O: float, Tb: float, C: float, M: float, 
                  exponents: Dict[str, float] = None, D_floor: float = 1.0) -> float:
        ex = exponents or {'alpha': 1.0, 'beta': 1.0, 'gamma': 1.0, 'delta': 1.0, 'epsilon': 1.0}
        d_val = (max(1.0, R)**ex['alpha'] * max(1.0, O)**ex['beta'] * max(1.0, Tb)**ex['gamma'] * max(1.0, C)**ex['delta'] * max(1.0, M)**ex['epsilon'])
        return max(D_floor, d_val)

    @staticmethod
    def alt1_additive(R: float, O: float, Tb: float, C: float, M: float, 
                      exponents: Dict[str, float] = None, D_floor: float = 1.0) -> float:
        ex = exponents or {'alpha': 1.0, 'beta': 1.0, 'gamma': 1.0, 'delta': 1.0, 'epsilon': 1.0}
        d_val = (ex['alpha']*max(1.0, R) + 
                 ex['beta']*max(1.0, O) + 
                 ex['gamma']*max(1.0, Tb) + 
                 ex['delta']*max(1.0, C) + 
                 ex['epsilon']*max(1.0, M) - 4.0)
        return max(D_floor, d_val)

class Freedom:
    F_MIN = 0.01

    @staticmethod
    def f1_scalar(P: float, D: float) -> float:
        return max(Freedom.F_MIN, P / max(1e-5, D))

    @staticmethod
    def f2_gradient(P_x: float, grad_D_x: np.ndarray) -> np.ndarray:
        return -P_x * grad_D_x

    @staticmethod
    def f3_discrete(P_list: np.ndarray, D_list: np.ndarray) -> int:
        F_list = P_list / np.maximum(1e-5, D_list)
        return int(np.argmax(F_list))

    @staticmethod
    def f4_swarm(P_list: np.ndarray, D_global: float) -> float:
        N = len(P_list)
        if N == 0: return Freedom.F_MIN
        return max(Freedom.F_MIN, np.sum(P_list) / (N * max(1e-5, D_global)))

    @staticmethod
    def f5_intelligent(P_ext: float, P_rec: float, D_ext: float, D_int: float) -> float:
        return max(Freedom.F_MIN, (P_ext * P_rec) / max(1e-5, D_ext * D_int))

class GlobalState:
    @staticmethod
    def f_global_harmonic(f_domains: np.ndarray, weights: np.ndarray) -> float:
        f_domains = np.maximum(Freedom.F_MIN, f_domains)
        weights = weights / np.sum(weights)
        harmonic_mean = 1.0 / np.sum(weights / f_domains)
        return max(Freedom.F_MIN, harmonic_mean)

    @staticmethod
    def regime_classifier(P_ext: float, P_rec: float) -> str:
        if P_ext <= 1.05 and P_rec <= 1.05: return "PASSIVE"
        elif P_rec <= 1.05: return "ACTIVE"
        else: return "INTELLIGENT"

class SwarmIntelligence:
    @staticmethod
    def s1_swarm_field(P_list: np.ndarray, N: int, D: float) -> float:
        if N == 0: return 0.01
        return np.sum(P_list) / (N * max(1e-5, D))

    @staticmethod
    def s2_stigmergy_derivative(kappa: float, tau: float, D: float) -> float:
        return kappa * tau / max(1e-5, D)

    @staticmethod
    def s3_explore_exploit_index(F_list: np.ndarray) -> float:
        mean_F = np.mean(F_list)
        if mean_F < 1e-5: return 0.0
        return np.var(F_list) / mean_F

    @staticmethod
    def s5_adaptive_learning_derivative(gamma: float, F_target: float, F_current: float, P_rec: float) -> float:
        return gamma * (F_target - F_current) * P_rec

    @staticmethod
    def s6_phase_transition(P_stig: float, D_threshold: float) -> bool:
        return P_stig > D_threshold
