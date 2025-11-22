# src/api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from src.models.detectors import MultiAnomalyDetector
from src.predictors import Forecaster

app = FastAPI(title="Energy AI - PZEM + Historic")

# Configurar CORS - MÁS ESPECÍFICO
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
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
# HANDLERS OPTIONS EXPLÍCITOS - CORREGIDOS
# -------------------------
@app.api_route("/detect", methods=["OPTIONS"])
async def options_detect():
    return JSONResponse(
        content={"message": "OK"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization"
        }
    )

@app.api_route("/detect_cut", methods=["OPTIONS"])
async def options_detect_cut():
    return JSONResponse(
        content={"message": "OK"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization"
        }
    )

@app.api_route("/detect_leak", methods=["OPTIONS"])
async def options_detect_leak():
    return JSONResponse(
        content={"message": "OK"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization"
        }
    )

@app.api_route("/predict_power", methods=["OPTIONS"])
async def options_predict_power():
    return JSONResponse(
        content={"message": "OK"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization"
        }
    )

@app.api_route("/predict_energy", methods=["OPTIONS"])
async def options_predict_energy():
    return JSONResponse(
        content={"message": "OK"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization"
        }
    )

@app.api_route("/advice", methods=["OPTIONS"])
async def options_advice():
    return JSONResponse(
        content={"message": "OK"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization"
        }
    )

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