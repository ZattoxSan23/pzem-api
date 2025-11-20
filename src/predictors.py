# src/predictors.py
import os, joblib, numpy as np, torch, torch.nn as nn
MODELS_DIR = "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMForecast(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_size,32), nn.ReLU(), nn.Linear(32,1))
    def forward(self,x):
        o,_ = self.lstm(x)
        o = o[:, -1, :]
        return self.fc(o)

class Forecaster:
    def __init__(self, which="power"):
        self.which = which
        if which=="power":
            self.scaler_path = os.path.join(MODELS_DIR,"scaler_power.pkl")
            self.model_path = os.path.join(MODELS_DIR,"lstm_power.pt")
        else:
            self.scaler_path = os.path.join(MODELS_DIR,"scaler_energy.pkl")
            self.model_path = os.path.join(MODELS_DIR,"lstm_energy.pt")
        if not (os.path.exists(self.scaler_path) and os.path.exists(self.model_path)):
            self.model = None
            print(f"Forecaster {which} missing models")
            return
        self.scaler = joblib.load(self.scaler_path)
        self.model = LSTMForecast().to(DEVICE)
        self.model.load_state_dict(torch.load(self.model_path, map_location=DEVICE))
        self.model.eval()

    def predict_next(self, recent_values):
        arr = np.array(recent_values[-30:], dtype=float).reshape(-1,1)
        if arr.shape[0] < 30:
            raise ValueError("Need at least 30 values")
        xs = self.scaler.transform(arr)
        x = torch.tensor(xs[np.newaxis,:,:], dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            y = self.model(x).cpu().numpy()[0][0]
        return float(self.scaler.inverse_transform([[y]])[0][0])
