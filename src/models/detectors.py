# src/models/detectors.py
import os, joblib, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from src.feature_engineering import FeatureEngineer

MODELS_DIR = "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, input_dim)
        )
    def forward(self,x):
        z = self.encoder(x)
        return self.decoder(z)

class MultiAnomalyDetector:
    def __init__(self, models_dir=MODELS_DIR):
        self.models_dir = models_dir
        self.fe = FeatureEngineer()
        self.iso = self._load_joblib("isolation_forest.pkl")
        self.scaler_auto = self._load_joblib("scaler_autoencoder.pkl")
        self.ae_train_mse = self._load_joblib("autoencoder_train_mse.pkl")
        self.ae = self._load_ae("autoencoder.pt")

    def _load_joblib(self, name):
        p = os.path.join(self.models_dir, name)
        return joblib.load(p) if os.path.exists(p) else None

    def _load_ae(self, name):
        p = os.path.join(self.models_dir, name)
        if os.path.exists(p) and self.scaler_auto is not None:
            input_dim = len(self.scaler_auto.mean_)
            hidden_dim = max(4, input_dim//2)
            model = AE(input_dim, hidden_dim).to(DEVICE)
            model.load_state_dict(torch.load(p, map_location=DEVICE))
            model.eval()
            return model
        return None

    # map PZEM/Firebase payload to dataset-like row
    def payload_to_row(self, payload: dict):
        row = {
            "Voltage": float(payload.get("voltage", payload.get("Voltage", 0.0))),
            "Current": float(payload.get("current", payload.get("Current", 0.0))),
            "Power": float(payload.get("power", payload.get("Power", 0.0))),
            "Frequency": float(payload.get("frequency", payload.get("Frequency", 0.0))),
            "Energy": float(payload.get("energy", payload.get("Energy", 0.0))),
            "Power_Factor": float(payload.get("powerFactor", payload.get("Power_Factor", 0.0))),
            "Date-Time": payload.get("timestamp", payload.get("Date-Time", None))
        }
        return row

    def detect_cut(self, payload: dict):
        v = float(payload.get("voltage", payload.get("Voltage", 0.0)))
        p = float(payload.get("power", payload.get("Power", 0.0)))
        if v <= 1.0 or p <= 0.05:
            return {"cut": True, "reason": "voltage or power near zero"}
        return {"cut": False}

    # -----------------------------------------------------
    # DETECCIÓN DE TENDENCIA → Fuga posible por aumento leve
    # -----------------------------------------------------
    def detect_trend_leak(self, recent_power_series, slope_threshold=0.01):
        if len(recent_power_series) < 5:
            return {"leak": False, "reason": "insufficient_data"}
        y = np.array(recent_power_series, dtype=float)
        x = np.arange(len(y))
        m, c = np.linalg.lstsq(
            np.vstack([x, np.ones(len(x))]).T,
            y, rcond=None
        )[0]
        leak = float(m) > slope_threshold
        return {"leak": bool(leak), "slope": float(m)}

    # -----------------------------------------------------
    # 🔥 FUNCIÓN QUE FALTABA → usada por /detect_leak
    # -----------------------------------------------------
    def detect_pzem_leak(self, series):
        trend = self.detect_trend_leak(series, slope_threshold=0.01)

        # criterio promedio mínimo
        if len(series) >= 5:
            avg = float(np.mean(series))
            if avg > 0.5:  # puedes modificar este threshold
                return {
                    "leak": True,
                    "method": "trend+avg",
                    "avg": avg,
                    "slope": trend.get("slope", 0)
                }

        return {
            "leak": False,
            "method": "trend",
            "slope": trend.get("slope", 0)
        }

    # -----------------------------------------------------

    def detect_iso(self, payload: dict):
        if self.iso is None:
            return {"isolation": "model_missing"}
        row = self.payload_to_row(payload)
        df = pd.DataFrame([row])
        df = self.fe.add_features(df)
        features = [
            "Power","Voltage","Current","Frequency","Power_Factor",
            "apparent","pf_eff","power_diff","power_roll_mean_3",
            "hour","dayofweek","is_weekend"
        ]
        X = df[features].values.astype(float)
        p = self.iso.predict(X)[0]
        return {"isolation": "anomaly" if p==-1 else "normal"}

    def detect_autoencoder(self, payload: dict):
        if self.ae is None or self.scaler_auto is None or self.ae_train_mse is None:
            return {"autoencoder": "model_missing"}
        row = self.payload_to_row(payload)
        df = pd.DataFrame([row])
        df = self.fe.add_features(df)
        features = [
            "Power","Voltage","Current","Frequency","Power_Factor",
            "apparent","pf_eff","power_diff","power_roll_mean_3",
            "hour","dayofweek","is_weekend"
        ]
        X = df[features].values.astype(float)
        Xs = self.scaler_auto.transform(X)
        xt = torch.tensor(Xs, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            recon = self.ae(xt).cpu().numpy()
        mse = float(np.mean((Xs - recon)**2, axis=1)[0])
        threshold = float(np.percentile(self.ae_train_mse, 95))
        return {
            "autoencoder": "anomaly" if mse > threshold else "normal",
            "mse": mse,
            "threshold": threshold
        }

    def advice(self, payload: dict):
        p = float(payload.get("power", payload.get("Power", 0.0)))
        v = float(payload.get("voltage", payload.get("Voltage", 0.0)))
        pf = float(payload.get("powerFactor", payload.get("Power_Factor", 0.0)))
        i = float(payload.get("current", payload.get("Current", 0.0)))

        tips = []
        if pf < 0.6:
            tips.append("Factor de potencia bajo — posible carga inductiva.")
        if p > 2000:
            tips.append("Consumo alto — revisa electrodomésticos grandes.")
        if v < 190 or v > 250:
            tips.append("Voltaje fuera de rango — problema de suministro.")
        if p <= 5 and i > 0.1:
            tips.append("Corriente presente con muy poca potencia → posible fuga.")
        if not tips:
            tips.append("Consumo y estado aparentemente normales.")

        return {"tips": tips}

    def detect_pzem(self, payload: dict):
        return {
            "cut": self.detect_cut(payload),
            "isolation": self.detect_iso(payload),
            "autoencoder": self.detect_autoencoder(payload),
            "advice": self.advice(payload)
        }
