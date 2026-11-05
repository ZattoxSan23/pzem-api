"""Validate telemetry history and make fallback behavior explicit."""
import math
from collections.abc import Mapping

WINDOW_SIZE = 30
MAX_HISTORY_SIZE = 10000


def normalize_history(history, metric):
    if metric not in ("power", "energy"):
        raise ValueError("Unsupported metric")
    if not isinstance(history, list) or not history:
        raise ValueError("history must be a non-empty list")
    if len(history) > MAX_HISTORY_SIZE:
        raise ValueError("history may contain at most 10000 readings")

    aliases = ("power", "P", "Power") if metric == "power" else ("energy", "Energy")
    values = []
    for index, item in enumerate(history):
        raw = item
        if isinstance(item, Mapping):
            key = next((key for key in aliases if key in item), None)
            if key is None:
                raise ValueError(f"history[{index}] is missing {metric}")
            raw = item[key]
        if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
            raise ValueError(f"history[{index}] must contain a number")
        try:
            value = float(raw)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(f"history[{index}] must contain a number") from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"history[{index}] must be finite and non-negative")
        values.append(value)
    return values


def forecast_history(history, forecaster, metric):
    values = normalize_history(history, metric)
    if forecaster.model is None or len(values) < WINDOW_SIZE:
        return values[-1], {
            "method": "last_observation",
            "reason": "model_unavailable" if forecaster.model is None else "insufficient_history",
            "samples": len(values),
            "required_samples": WINDOW_SIZE,
        }

    prediction = float(forecaster.predict_next(values))
    if not math.isfinite(prediction) or prediction < 0:
        raise RuntimeError("Model returned an invalid prediction")
    return prediction, {"method": "lstm", "samples": len(values)}
