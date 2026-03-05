import numpy as np
from collections import deque
from typing import Dict, Any, Tuple

class EdgeFusionPipeline:
    def __init__(self, window_size: int = 10080):
        self.window_size = window_size
        self.history = {
            'temperature': deque(maxlen=window_size),
            'co2': deque(maxlen=window_size),
            'light': deque(maxlen=window_size),
            'occupancy': deque(maxlen=window_size)
        }
    
    def process_reading(self, raw_data: Dict[str, float], P_tier: float = 1.0) -> Tuple[Dict[str, Any], list]:
        alerts = []
        processed = {}
        
        for sensor, value in raw_data.items():
            if sensor not in self.history: continue
            
            is_outlier = False
            if len(self.history[sensor]) > 10:
                mu = np.mean(self.history[sensor])
                sigma = np.std(self.history[sensor])
                if sigma > 0 and abs(value - mu) > 4 * sigma:
                    is_outlier = True
                    alerts.append(f"[OUTLIER] {sensor.upper()} reading {value} exceeds mu±4sigma")
            
            self.history[sensor].append(value)
            processed[sensor] = {'value': value, 'is_outlier': is_outlier}
            
        state = {}
        
        if 'temperature' in processed:
            D_thermal = 1.0 + abs(processed['temperature']['value'] - 22.0) / 2.0
            F_thermal = max(0.01, P_tier / D_thermal)
            conf = 0.50 if processed['temperature']['is_outlier'] else 0.95
            state['thermal'] = {'P': P_tier, 'D': D_thermal, 'F': F_thermal, 'confidence': conf}
            
        if 'co2' in processed:
            D_air = 1.0 + max(0, processed['co2']['value'] - 420.0) / 400.0
            F_air = max(0.01, P_tier / D_air)
            conf = 0.50 if processed['co2']['is_outlier'] else 0.95
            state['air'] = {'P': P_tier, 'D': D_air, 'F': F_air, 'confidence': conf}
            
            if F_air < 0.50:
                alerts.append(f"[CASCADE RISK] F_air critical drop to {F_air:.2f}")

        return state, alerts

class SensorPlacementOptimizer:
    def __init__(self, width: float = 30.0, height: float = 18.0, resolution: float = 1.0):
        self.width = width
        self.height = height
        self.res = resolution
        
        x_coords = np.arange(0, width, resolution)
        y_coords = np.arange(0, height, resolution)
        self.grid_x, self.grid_y = np.meshgrid(x_coords, y_coords)
        self.grid_points = np.vstack([self.grid_x.ravel(), self.grid_y.ravel()]).T

    def evaluate_coverage(self, node_positions: np.ndarray, radius: float) -> float:
        if len(node_positions) == 0: return 0.0
        diff = self.grid_points[:, np.newaxis, :] - node_positions[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        min_dists = np.min(sq_dists, axis=1)
        covered_mask = min_dists <= (radius**2)
        return float(np.sum(covered_mask) / len(self.grid_points))
