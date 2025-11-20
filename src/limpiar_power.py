# limpiar_power.py

import pandas as pd

class PowerCleaner:
    def __init__(self, filepath: str):
        self.filepath = filepath

    def load(self):
        print("📥 Cargando dataset...")
        df = pd.read_csv(self.filepath, sep=';', low_memory=False)
        print("✔ Dataset cargado")
        return df

    def clean(self, df: pd.DataFrame):
        print("🧹 Iniciando limpieza...")

        # Crear Datetime
        df["Datetime"] = pd.to_datetime(
            df["Date"] + " " + df["Time"],
            format="%d/%m/%Y %H:%M:%S",
            errors="coerce"
        )
        df = df.drop(columns=["Date", "Time"])
        df = df.dropna(subset=["Datetime"])

        # Limpiar cada columna no datetime
        for c in df.columns:
            if c == "Datetime":
                continue

            # Convertir a string primero para poder usar .str
            df[c] = df[c].astype(str)

            # Reemplazar comas y valores raros
            df[c] = (
                df[c]
                .str.replace(",", ".", regex=False)
                .str.replace("?", "0", regex=False)
                .str.strip()
            )

            # Convertir a número
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # Eliminar filas malas
        df = df.dropna()

        df = df.sort_values("Datetime").reset_index(drop=True)

        print("✔ Limpieza terminada")
        return df

    def save(self, df: pd.DataFrame, output_path: str):
        print(f"💾 Guardando CSV limpio en: {output_path}")
        df.to_csv(output_path, index=False)
        print("✔ Guardado")

if __name__ == "__main__":
    cleaner = PowerCleaner("data/household_power_consumption.txt")  # ← RUTA CORRECTA
    df = cleaner.load()
    df = cleaner.clean(df)
    cleaner.save(df, "data/clean_power.csv")  # ← RUTA CORRECTA
