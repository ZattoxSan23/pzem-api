# Energy AI — API de análisis PZEM

API FastAPI para analizar lecturas eléctricas y ejecutar modelos de detección de anomalías y predicción. Incluye artefactos de Isolation Forest, autoencoder y LSTM en `models/`. El backend de adquisición ESP32 se mantiene en el repositorio `Backend`.

## Instalación y ejecución

```sh
python -m venv .venv
# Activar .venv según tu sistema operativo
python -m pip install -r requirements.txt
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Ejecuta desde la raíz del repositorio. Los modelos se resuelven respecto al código, no al directorio de trabajo. Consulta `/docs` para ver el contrato HTTP y `/health` para comprobar el proceso.

## Entradas y resultados

| Ruta | Entrada | Resultado |
| --- | --- | --- |
| `POST /analyze` | voltaje, corriente, potencia, energía, frecuencia y factor de potencia | detecciones de corte/anomalía |
| `POST /detect_cut` | lectura PZEM | heurística de corte |
| `POST /detect_leak` | `{"history": [10, 11, 10, 10, 11]}` | heurística de consumo constante bajo |
| `POST /predict_power` | historial numérico o lecturas con `power`, `P` o `Power` | `next_power_w` y método |
| `POST /predict_energy` | historial numérico o lecturas con `energy` o `Energy` | `next_energy` y método |
| `POST /advice` | lectura PZEM | mensajes basados en umbrales |

El historial debe estar ordenado cronológicamente. No se inventan timestamps ni se reordenan muestras. Solo se aceptan valores finitos no negativos; máximo 10 000 muestras por solicitud.

El LSTM usa las últimas **30 muestras**. Con menos datos o sin modelo se devuelve la última observación y `method: "last_observation"`, junto con la razón. Un fallo del modelo devuelve 503; los datos inválidos, 422. El respaldo no equivale a una predicción entrenada ni aplica incrementos arbitrarios.

## Pruebas

```sh
python -m unittest discover -s tests -v
python -m compileall -q src tests
```

Las pruebas del historial usan dobles de modelo y biblioteca estándar; no requieren GPU ni descargar dependencias ML. Cubren ventanas de 29/30 muestras, campos ausentes, NaN, infinito, valores negativos y resultados inválidos.

## Límites conocidos

Prototipo: no hay métricas publicadas de precisión ni validación independiente de los artefactos. Las reglas de fugas/cortes son heurísticas y pueden confundir cargas apagadas o consumo en reposo. Faltan versionado de datasets y dependencias reproducibles, autenticación, límites globales de solicitudes y configuración de CORS por entorno. La carga de modelos ocurre al importar la aplicación; un artefacto incompatible puede impedir el arranque. Los archivos pickle y PyTorch deben provenir de fuentes confiables.
