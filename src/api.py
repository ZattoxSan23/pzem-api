from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime
import numpy as np

from src.models.detectors import MultiAnomalyDetector
from src.predictors import Forecaster

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
        
        # Convertir a lista de números
        if isinstance(history[0], dict):
            power_values = [float(h.get("power", h.get("P", 0))) for h in history]
        else:
            power_values = [float(x) for x in history]
            
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
            
    except Exception as e:
        raise HTTPException(500, f"Leak detection error: {str(e)}")

@app.post("/predict_energy")
async def predict_energy(data: HistoryList):
    """Predicción de consumo de energía"""
    try:
        if forecaster_energy.model is None:
            # Predicción simple si el modelo no está cargado
            if data.history:
                last_value = float(data.history[-1]) if data.history else 0
                return {"next_energy": last_value * 1.02}  # Incremento del 2%
            return {"next_energy": 0.0}
            
        history = [float(x) for x in data.history]
        if len(history) < 10:
            return {"next_energy": history[-1] if history else 0.0}
            
        prediction = forecaster_energy.predict_next(history)
        return {"next_energy": prediction}
        
    except Exception as e:
        # Fallback a predicción simple
        if data.history:
            last_value = float(data.history[-1]) if data.history else 0
            return {"next_energy": last_value}
        return {"next_energy": 0.0}

@app.post("/predict_power")
async def predict_power(data: HistoryList):
    """Predicción de potencia"""
    try:
        if forecaster_power.model is None:
            # Predicción simple si el modelo no está cargado
            if data.history:
                last_value = float(data.history[-1]) if data.history else 0
                return {"next_power_w": last_value}
            return {"next_power_w": 0.0}
            
        history = [float(x) for x in data.history]
        if len(history) < 10:
            return {"next_power_w": history[-1] if history else 0.0}
            
        prediction = forecaster_power.predict_next(history)
        return {"next_power_w": prediction}
        
    except Exception as e:
        # Fallback
        if data.history:
            last_value = float(data.history[-1]) if data.history else 0
            return {"next_power_w": last_value}
        return {"next_power_w": 0.0}

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