# src/train_models.py
import os, joblib, numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.feature_engineering import FeatureEngineer
from src.models.detectors import AE
from src.predictors import LSTMForecast

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_csv(path="data/DPU-ALDOSKI.csv"):
    # read, strip quotes in header automatically
    df = pd.read_csv(path)
    # normalize columns
    df.columns = [c.strip().strip('"') for c in df.columns]
    # prefer 'Date-Time' as timestamp
    if 'Date-Time' in df.columns:
        # remove am/pm trailing to help parsing if needed
        try:
            df['Date-Time'] = df['Date-Time'].str.replace('am','').str.replace('pm','', regex=False)
        except Exception:
            pass
    return df

def make_features(df):
    fe = FeatureEngineer()
    df2 = fe.add_features(df)
    features = ["Power","Voltage","Current","Frequency","Power_Factor","apparent","pf_eff","power_diff","power_roll_mean_3","hour","dayofweek","is_weekend"]
    X = df2[features].values.astype(float)
    return df2, X, features

def train_isolation(X):
    iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=42)
    iso.fit(X)
    joblib.dump(iso, os.path.join(MODELS_DIR,"isolation_forest.pkl"))
    print("isolation saved")

def train_autoencoder(X, epochs=30, batch_size=128):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    joblib.dump(scaler, os.path.join(MODELS_DIR,"scaler_autoencoder.pkl"))
    Xt = torch.tensor(Xs, dtype=torch.float32)
    ds = TensorDataset(Xt)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    input_dim = Xs.shape[1]
    hidden_dim = max(4, input_dim//2)
    model = AE(input_dim, hidden_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for ep in range(1, epochs+1):
        epoch_loss = 0.0
        for (b,) in loader:
            b = b.to(DEVICE)
            opt.zero_grad()
            out = model(b)
            loss = loss_fn(out,b)
            loss.backward(); opt.step()
            epoch_loss += loss.item()*b.size(0)
        epoch_loss /= len(ds)
        if ep%5==0 or ep==1:
            print("AE ep",ep,"loss",epoch_loss)
    torch.save(model.state_dict(), os.path.join(MODELS_DIR,"autoencoder.pt"))
    model.eval()
    with torch.no_grad():
        recon = model(Xt.to(DEVICE)).cpu().numpy()
    mse = np.mean((Xs-recon)**2, axis=1)
    joblib.dump(mse, os.path.join(MODELS_DIR,"autoencoder_train_mse.pkl"))
    print("autoencoder saved")

def train_lstm(series, scaler_out, model_out, epochs=30, window=30):
    scaler = MinMaxScaler()
    vals = series.reshape(-1,1)
    vs = scaler.fit_transform(vals)
    joblib.dump(scaler, scaler_out)
    Xs, ys = [], []
    for i in range(len(vs)-window):
        Xs.append(vs[i:i+window])
        ys.append(vs[i+window])
    Xs = np.array(Xs).astype(np.float32)
    ys = np.array(ys).astype(np.float32)
    ds = TensorDataset(torch.tensor(Xs), torch.tensor(ys))
    loader = DataLoader(ds, batch_size=256, shuffle=True)
    model = LSTMForecast().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for ep in range(1, epochs+1):
        epoch_loss = 0.0
        for xb,yb in loader:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out,yb)
            loss.backward(); opt.step()
            epoch_loss += loss.item()*xb.size(0)
        epoch_loss /= len(ds)
        if ep%5==0 or ep==1:
            print("LSTM ep",ep,"loss",epoch_loss)
    torch.save(model.state_dict(), model_out)
    print("lstm saved", model_out)

if __name__ == "__main__":
    df = load_csv("data/DPU-ALDOSKI.csv")
    df2, X, features = make_features(df)
    train_isolation(X)
    train_autoencoder(X, epochs=30)
    # train LSTM for power (Power already in W)
    power_w = df2["Power"].values.astype(float)
    train_lstm(power_w, os.path.join(MODELS_DIR,"scaler_power.pkl"), os.path.join(MODELS_DIR,"lstm_power.pt"), epochs=30)
    # train LSTM for energy if present (Energy column)
    if "Energy" in df2.columns:
        energy_vals = df2["Energy"].values.astype(float)
        train_lstm(energy_vals, os.path.join(MODELS_DIR,"scaler_energy.pkl"), os.path.join(MODELS_DIR,"lstm_energy.pt"), epochs=30)
    else:
        print("Energy column not found, skipping energy LSTM")
