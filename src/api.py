# src/api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Agregar esta importación
from pydantic import BaseModel
from src.models.detectors import MultiAnomalyDetector
from src.predictors import Forecaster

app = FastAPI(title="Energy AI - PZEM + Historic")

# Configurar CORS - AGREGAR ESTO
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especifica los dominios permitidos
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos incluyendo OPTIONS
    allow_headers=["*"],
)

detector = MultiAnomalyDetector()
forecaster_power = Forecaster(which="power")
forecaster_energy = Forecaster(which="energy")

# -------------------------
# PAYLOAD DEL PZEM
# -------------------------
class PZEMPayload(BaseModel):
    voltage: float
    current: float
    power: float
    energy: float = 0.0
    frequency: float = 0.0
    powerFactor: float = 0.0
    timestamp: str = None


# -------------------------
# MODELO PARA HISTORIA (FIX)
# -------------------------
class HistoryList(BaseModel):
    history: list


@app.get("/")
def root():
    return {"status": "running"}


# -------------------------
# DETECCIÓN ANOMALÍAS
# -------------------------
@app.post("/detect")
def detect(payload: PZEMPayload):
    return detector.detect_pzem(payload.dict())

@app.post("/detect_cut")
def detect_cut(payload: PZEMPayload):
    return detector.detect_cut(payload.dict())


# -------------------------
# DETECCIÓN DE FUGAS / LEAK FIX + Pydantic
# -------------------------
@app.post("/detect_leak")
def detect_leak(data: HistoryList):
    history = data.history

    if len(history) == 0:
        raise HTTPException(400, "history required")

    # Acepta lista de números o lista de dicts
    if isinstance(history[0], dict):
        series = [float(h.get("power", 0)) for h in history]
    else:
        series = [float(x) for x in history]

    return detector.detect_pzem_leak(series)


# -------------------------
# PREDICCIÓN
# -------------------------
@app.post("/predict_power")
def predict_power(data: HistoryList):
    if forecaster_power.model is None:
        raise HTTPException(503, "power forecaster not trained")

    return {"next_power_w": forecaster_power.predict_next(data.history)}


@app.post("/predict_energy")
def predict_energy(data: HistoryList):
    if forecaster_energy.model is None:
        raise HTTPException(503, "energy forecaster not trained")

    return {"next_energy": forecaster_energy.predict_next(data.history)}


# -------------------------
# CONSEJOS / ADVICE
# -------------------------
@app.post("/advice")
def advice(payload: PZEMPayload):
    return detector.advice(payload.dict())