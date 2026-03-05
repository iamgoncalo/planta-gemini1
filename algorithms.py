import numpy as np

class AFIOptimizer:
    def __init__(self, objective_fn=None):
        self.objective_fn = objective_fn

class A1_GreedyZoneSelector(AFIOptimizer):
    def optimize(self, zones):
        best_zone, best_f = None, -1.0
        for zone in zones:
            F = zone.get('P', 1.0) / max(1e-5, zone.get('D', 1.0))
            if F > best_f:
                best_f, best_zone = F, zone
        return best_zone, best_f
