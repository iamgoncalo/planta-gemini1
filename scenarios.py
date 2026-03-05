import numpy as np
import pandas as pd

class ScenarioMatrix:
    def __init__(self):
        np.random.seed(42)

    def f_comfort_from_temp(self, T: float) -> float:
        return np.clip(1.0 - np.abs(T - 22.0) / 10.0, 0.01, 1.0)

    def run_sc01_morning_preheat(self) -> pd.DataFrame:
        time_mins = np.arange(180)
        
        T_i1 = np.ones(180) * 14.0
        T_i1[120:] = 14.0 + (np.arange(60) / 60.0) * 8.0 
        
        T_i3 = np.ones(180) * 14.0
        T_i3[60:120] = 14.0 + (np.arange(60) / 60.0) * 8.0 
        T_i3[120:] = 22.0
        
        return pd.DataFrame({
            'Time_Min': time_mins, 'T_I1': T_i1, 'T_I3': T_i3,
            'F_I1': self.f_comfort_from_temp(T_i1), 'F_I3': self.f_comfort_from_temp(T_i3)
        })

    def run_sc04_co2_cascade(self) -> tuple:
        time_mins = np.arange(120)
        CO2_base = 400.0 + (time_mins ** 1.5) * 1.5 
        F_i1 = np.clip(1.0 - np.maximum(0, CO2_base - 800) / 1000.0, 0.01, 1.0)
        
        CO2_i3 = CO2_base.copy()
        CO2_i3[30:] = CO2_base[30] - (np.arange(90) * 2.0)
        CO2_i3 = np.maximum(400, CO2_i3)
        F_i3 = np.clip(1.0 - np.maximum(0, CO2_i3 - 800) / 1000.0, 0.01, 1.0)
        
        i1_drop_time = np.argmax(F_i1 < 0.5) if np.any(F_i1 < 0.5) else 120
        i3_detect_time = 30
        
        df = pd.DataFrame({'Time_Min': time_mins, 'F_I1': F_i1, 'F_I3': F_i3, 'CO2_I1': CO2_base, 'CO2_I3': CO2_i3})
        return df, i1_drop_time, i3_detect_time

    def run_sc06_annual_baseline(self) -> pd.DataFrame:
        months = np.arange(1, 13)
        F_i1 = np.random.uniform(0.5, 0.7, 12)
        F_i3 = np.random.uniform(0.75, 0.95, 12)
        return pd.DataFrame({'Month': months, 'F_I1': F_i1, 'F_I3': F_i3})
