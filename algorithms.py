import numpy as np
import networkx as nx
import heapq
from typing import List, Tuple, Callable, Dict, Any

class AFIOptimizer:
    """Base class for all PlantaOS F-Field algorithms."""
    def __init__(self, objective_fn: Callable = None):
        self.objective_fn = objective_fn

    def optimize(self, problem: Any) -> Tuple[Any, float]:
        raise NotImplementedError

class A1_GreedyZoneSelector(AFIOptimizer):
    """A1: Greedy Zone Selector. Rule: i* = argmax_i (P_i / D_i)"""
    def optimize(self, zones: List[Dict[str, float]]) -> Tuple[Dict[str, float], float]:
        best_zone, best_f = None, -1.0
        for zone in zones:
            F = zone.get('P', 1.0) / max(1e-5, zone.get('D', 1.0))
            if F > best_f:
                best_f, best_zone = F, zone
        return best_zone, best_f

class A2_GradientDescentFField(AFIOptimizer):
    """A2: Gradient Descent on F-Field. Rule: dx/dt = -P(x) * grad_D(x)"""
    def optimize(self, start_x: np.ndarray, grad_D_fn: Callable, P_fn: Callable, 
                 lr: float = 0.1, max_steps: int = 100, tol: float = 1e-4) -> Tuple[np.ndarray, int]:
        x = np.copy(start_x)
        for step in range(max_steps):
            dx = -P_fn(x) * grad_D_fn(x) * lr
            if np.linalg.norm(dx) < tol:
                return x, step
            x += dx
        return x, max_steps

class A4_AFI_AStar(AFIOptimizer):
    """A4: AFI-Weighted A* Pathfinding for Evacuation Routing."""
    def optimize(self, G: nx.Graph, start: str, goal: str, 
                 P_dict: Dict[str, float], D_dict: Dict[str, float]) -> Tuple[List[str], float]:
        open_set = []
        heapq.heappush(open_set, (0.0, start, [start]))
        g_score = {node: float('inf') for node in G.nodes}
        g_score[start] = 0.0
        
        while open_set:
            _, current, path = heapq.heappop(open_set)
            if current == goal:
                min_f = min([P_dict.get(n, 1.0) / max(1e-5, D_dict.get(n, 1.0)) for n in path])
                return path, min_f
                
            for neighbor in G.neighbors(current):
                edge_weight = G[current][neighbor].get('weight', 1.0)
                tentative_g = g_score[current] + (edge_weight / P_dict.get(current, 1.0))
                
                if tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + (1.0 * D_dict.get(neighbor, 1.0))
                    heapq.heappush(open_set, (f_score, neighbor, path + [neighbor]))
        return [], 0.0

class B1_ACO_AFI(AFIOptimizer):
    """B1: Ant Colony Optimization via Stigmergy."""
    def __init__(self, alpha=1.0, beta=2.0, rho=0.1, n_ants=20, n_iter=100):
        self.alpha, self.beta, self.rho, self.n_ants, self.n_iter = alpha, beta, rho, n_ants, n_iter

    def optimize(self, D_matrix: np.ndarray, P_matrix: np.ndarray = None):
        n_nodes = len(D_matrix)
        P_matrix = np.ones((n_nodes, n_nodes)) if P_matrix is None else P_matrix
        tau = np.ones((n_nodes, n_nodes))
        best_path, best_D = None, float('inf')

        for _ in range(self.n_iter):
            paths, path_Ds = [], []
            for _ in range(self.n_ants):
                curr = np.random.randint(n_nodes)
                path, visited, path_D = [curr], {curr}, 0.0
                for _ in range(n_nodes - 1):
                    probs = np.zeros(n_nodes)
                    for j in range(n_nodes):
                        if j not in visited:
                            F_ij = P_matrix[curr][j] / max(1e-5, D_matrix[curr][j])
                            probs[j] = (tau[curr][j] ** self.alpha) * (F_ij ** self.beta)
                    
                    if np.sum(probs) == 0:
                        next_node = np.random.choice([u for u in range(n_nodes) if u not in visited])
                    else:
                        next_node = np.random.choice(range(n_nodes), p=probs/np.sum(probs))
                        
                    path.append(next_node)
                    visited.add(next_node)
                    path_D += D_matrix[curr][next_node]
                    curr = next_node
                
                path_D += D_matrix[curr][path[0]]
                paths.append(path)
                path_Ds.append(path_D)
                if path_D < best_D: 
                    best_D, best_path = path_D, path

            tau *= (1.0 - self.rho)
            for path, d_val in zip(paths, path_Ds):
                F_path = 1.0 / max(1e-5, d_val)
                for i in range(n_nodes):
                    u, v = path[i], path[(i + 1) % n_nodes]
                    tau[u][v] += F_path
                    tau[v][u] += F_path
        return best_path, best_D

class B2_PSO_AFI(AFIOptimizer):
    """B2: Particle Swarm Optimization scaled by local F."""
    def __init__(self, n_particles=30, n_iter=100):
        self.n_particles, self.n_iter = n_particles, n_iter

    def optimize(self, D_func, bounds):
        dim = len(bounds)
        lower, upper = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])
        x = np.random.uniform(lower, upper, (self.n_particles, dim))
        v, pbest = np.zeros((self.n_particles, dim)), np.copy(x)
        pbest_D = np.array([D_func(p) for p in x])
        gbest_idx = np.argmin(pbest_D)
        gbest, gbest_D = np.copy(pbest[gbest_idx]), pbest_D[gbest_idx]

        for it in range(self.n_iter):
            w = 0.7 - (0.3) * (it / self.n_iter)
            r1, r2 = np.random.rand(self.n_particles, dim), np.random.rand(self.n_particles, dim)
            for i in range(self.n_particles):
                F_personal = 1.0 / max(1e-5, pbest_D[i])
                F_global = 1.0 / max(1e-5, gbest_D)
                v[i] = w * v[i] + 1.5 * F_personal * r1[i] * (pbest[i] - x[i]) + 1.5 * F_global * r2[i] * (gbest - x[i])
                x[i] = np.clip(x[i] + v[i], lower, upper)
                
                current_D = D_func(x[i])
                if current_D < pbest_D[i]:
                    pbest[i], pbest_D[i] = np.copy(x[i]), current_D
                    if current_D < gbest_D: 
                        gbest, gbest_D = np.copy(x[i]), current_D
        return gbest, gbest_D
