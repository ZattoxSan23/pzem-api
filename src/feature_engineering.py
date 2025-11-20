# src/feature_engineering.py
import pandas as pd
import numpy as np

class FeatureEngineer:
    """
    Feature engineering adapted to DPU-ALDOSKI.csv
    Expects columns: Voltage, Current, Power, Frequency, Energy, Power_Factor, Date-Time
    Produces Datetime, hour, dayofweek, is_weekend, power_diff, power_roll_mean_3, apparent, pf_eff
    """
    def add_features(self, df: pd.DataFrame):
        df = df.copy()

        # Normalize column names (strip quotes if present)
        df.columns = [c.strip().strip('"') for c in df.columns]

        # parse datetime: prefer 'Date-Time', fallback to common names
        if 'Date-Time' in df.columns:
            df['Datetime'] = pd.to_datetime(df['Date-Time'].str.replace(r'\s?am$', '', regex=True).str.replace(r'\s?pm$', '', regex=True), errors='coerce')
            # Attempt also to parse with am/pm if necessary
            try:
                df['Datetime'] = pd.to_datetime(df['Date-Time'], errors='coerce')
            except Exception:
                pass
        elif 'Date_Time' in df.columns:
            df['Datetime'] = pd.to_datetime(df['Date_Time'], errors='coerce')
        elif 'timestamp' in df.columns:
            df['Datetime'] = pd.to_datetime(df['timestamp'], errors='coerce')
        else:
            df['Datetime'] = pd.to_datetime(df.index, errors='coerce')

        # Ensure numeric columns
        for col in ['Voltage','Current','Power','Frequency','Energy','Power_Factor']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            else:
                df[col] = 0.0

        # derived features
        df['apparent'] = df['Voltage'] * df['Current']     # V * A
        # power in dataset is in W (looks like). Keep as 'Power' (W)
        df['power_diff'] = df['Power'].diff().fillna(0.0)
        df['power_roll_mean_3'] = df['Power'].rolling(window=3, min_periods=1).mean()
        df['pf_eff'] = df['Power_Factor'].replace(0, np.nan)
        df['pf_eff'] = df['pf_eff'].fillna((df['Power']/(df['apparent']+1e-9))).fillna(0.0)
        # time features
        df['hour'] = df['Datetime'].dt.hour.fillna(0).astype(int)
        df['dayofweek'] = df['Datetime'].dt.dayofweek.fillna(0).astype(int)
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

        # fill remaining nans
        df = df.ffill().fillna(0.0)

        return df
