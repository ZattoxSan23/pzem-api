from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime
import numpy as np

from src.models.detectors import MultiAnomalyDetector
from src.predictors import Forecaster
from src.history import normalize_history, forecast_history

app = FastAPI(title="Energy AI - PZEM + Historic")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
)

# Inicializar componentes
detector = MultiAnomalyDetector()
forecaster_power = Forecaster(which="power")
forecaster_energy = Forecaster(which="energy")

# -------------------------
# MODELOS PYDANTIC
# -------------------------
class PZEMPayload(BaseModel):
    voltage: float = 0.0
    current: float = 0.0
    power: float = 0.0
    energy: float = 0.0
    frequency: float = 0.0
    powerFactor: float = 0.0
    timestamp: str = None

class HistoryList(BaseModel):
    history: list

class EnergyAnalysisRequest(BaseModel):
    voltage: float = 0.0
    current: float = 0.0
    power: float = 0.0
    energy: float = 0.0
    frequency: float = 0.0
    powerFactor: float = 0.0

# -------------------------
# ENDPOINTS BÁSICOS
# -------------------------
@app.get("/")
def root():
    return {"status": "running", "service": "Energy AI API"}

@app.get("/health")
async def health_check():
    """Endpoint para verificar el estado del API"""
    return {
        "status": "healthy", 
        "message": "Energy AI API is running",
        "timestamp": datetime.now().isoformat()
    }

# -------------------------
# HANDLERS OPTIONS
# -------------------------
@app.api_route("/{path:path}", methods=["OPTIONS"])
async def options_handler(path: str):
    return JSONResponse(
        content={"message": "OK"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Requested-With"
        }
    )

# -------------------------
# ANÁLISIS DE ENERGÍA - ENDPOINTS PRINCIPALES
# -------------------------
@app.post("/analyze")
async def analyze_energy(payload: PZEMPayload):
    """Análisis completo de energía"""
    try:
        data = payload.dict()
        result = detector.detect_pzem(data)
        return result
    except Exception as e:
        raise HTTPException(500, f"Analysis error: {str(e)}")

@app.post("/detect_cut")
async def detect_cut(payload: PZEMPayload):
    """Detección de cortes de energía"""
    try:
        data = payload.dict()
        result = detector.detect_cut(data)
        return result
    except Exception as e:
        raise HTTPException(500, f"Cut detection error: {str(e)}")

@app.post("/detect_leak")
async def detect_leak(data: HistoryList):
    """Detección de fugas de energía"""
    try:
        history = data.history
        
        if not history or len(history) == 0:
            return {"leak": False, "reason": "No data provided"}
        
        power_values = normalize_history(history, "power")
            
        # Lógica simple de detección de fugas
        if len(power_values) < 5:
            return {"leak": False, "reason": "Insufficient data for leak detection"}
            
        avg_power = np.mean(power_values)
        std_power = np.std(power_values)
        
        # Si hay consumo constante bajo, posible fuga
        if avg_power > 0 and std_power < avg_power * 0.1 and avg_power < 50:
            return {
                "leak": True, 
                "reason": f"Possible energy leak detected: constant {avg_power:.2f}W consumption"
            }
        else:
            return {"leak": False, "reason": "No energy leak detected"}
            
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Leak detection failed") from exc

def predict_response(data, forecaster, metric, response_key):
    try:
        prediction, metadata = forecast_history(data.history, forecaster, metric)
        return {response_key: prediction, **metadata}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        # Do not disguise a failed model as a successful AI prediction.
        raise HTTPException(status_code=503, detail="Prediction model unavailable") from exc


@app.post("/predict_energy")
def predict_energy(data: HistoryList):
    return predict_response(data, forecaster_energy, "energy", "next_energy")


@app.post("/predict_power")
def predict_power(data: HistoryList):
    return predict_response(data, forecaster_power, "power", "next_power_w")


@app.post("/advice")
async def get_energy_advice(payload: PZEMPayload):
    """Consejos de eficiencia energética"""
    try:
        data = payload.dict()
        voltage = data.get('voltage', 0)
        current = data.get('current', 0) 
        power = data.get('power', 0)
        power_factor = data.get('powerFactor', 0)
        
        tips = []
        
        # Análisis de voltaje
        if voltage > 0:
            if voltage < 180:
                tips.append("🔴 VOLTAJE PELIGROSAMENTE BAJO: Puede dañar equipos eléctricos")
            elif voltage > 250:
                tips.append("🔴 VOLTAJE PELIGROSAMENTE ALTO: Desconecta equipos sensibles")
            elif 210 <= voltage <= 230:
                tips.append("✅ Voltaje óptimo para Perú")
            else:
                tips.append("🟡 Voltaje fuera del rango ideal pero aceptable")
        
        # Análisis de potencia
        if power > 0:
            if power > 2000:
                tips.append("⚡ ALTO CONSUMO: Equipo de alta potencia como aire acondicionado o calefactor")
            elif power > 1000:
                tips.append("💡 CONSUMO MODERADO-ALTO: Posiblemente refrigeradora o lavadora")
            elif power < 100:
                tips.append("💡 CONSUMO EFICIENTE: Equipo de bajo consumo")
        
        # Análisis de factor de potencia
        if power_factor > 0:
            if power_factor < 0.7:
                tips.append("🟡 FACTOR DE POTENCIA BAJO: Normal en motores, considera compensación")
            elif power_factor >= 0.9:
                tips.append("✅ Factor de potencia excelente")
        
        # Consejos generales
        if not tips:
            if power > 0:
                tips.append("✅ Consumo normal dentro de parámetros esperados")
            else:
                tips.append("💡 Equipo apagado o en standby")
        
        return {"tips": tips[:3]}  # Máximo 3 consejos
        
    except Exception as e:
        return {"tips": ["💡 Monitoreando consumo energético"]}

# -------------------------
# MANEJO DE ERRORES GLOBAL
# -------------------------
@app.exception_handler(500)
async def internal_server_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "message": str(exc)}
    )

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "message": "The requested endpoint does not exist"}
    )