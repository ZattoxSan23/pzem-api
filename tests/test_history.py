import unittest
from src.history import normalize_history, forecast_history


class FakeForecaster:
    def __init__(self, available=True, result=42.0):
        self.model = object() if available else None
        self.result = result
        self.called = False

    def predict_next(self, values):
        self.called = True
        return self.result


class HistoryTests(unittest.TestCase):
    def test_numeric_and_device_readings(self):
        self.assertEqual(normalize_history([1, "2.5", {"P": 3}], "power"), [1, 2.5, 3])
        self.assertEqual(normalize_history([{"energy": 0}], "energy"), [0])

    def test_rejects_invalid_input(self):
        for history in ([], [True], [None], [-1], ["NaN"], ["Infinity"],
                        [{}], [{"energy": 2}], ["bad"], [1] * 10001):
            with self.subTest(history=str(history)[:40]):
                with self.assertRaises(ValueError):
                    normalize_history(history, "power")

    def test_short_window_is_explicit_baseline(self):
        for length in (1, 10, 29):
            model = FakeForecaster()
            value, info = forecast_history(list(range(length)), model, "power")
            self.assertEqual(value, length - 1)
            self.assertEqual(info["reason"], "insufficient_history")
            self.assertFalse(model.called)

    def test_30_samples_use_model(self):
        model = FakeForecaster(result=7)
        value, info = forecast_history([1] * 30, model, "energy")
        self.assertEqual(value, 7)
        self.assertEqual(info["method"], "lstm")
        self.assertTrue(model.called)

    def test_missing_model_does_not_invent_growth(self):
        value, info = forecast_history([10] * 30, FakeForecaster(False), "energy")
        self.assertEqual(value, 10)
        self.assertEqual(info["reason"], "model_unavailable")

    def test_invalid_model_output_is_not_success(self):
        for result in (float("nan"), float("inf"), -1):
            with self.assertRaises(RuntimeError):
                forecast_history([1] * 30, FakeForecaster(result=result), "power")


if __name__ == "__main__":
    unittest.main()
