# predict_vol_qqq_5m.py
# Predice volatilidad futura del QQQ en granularidad 5m usando tu data_cleaner.py (Alpaca / yfinance / CSV)

from pathlib import Path
import sys
import pandas as pd
import numpy as np

# --- Ajusta el import de src/ para que funcione al ejecutar el .py ---
AQUI = Path(__file__).resolve().parent
SRC = (AQUI / "src").resolve()
if SRC.exists():
    sys.path.insert(0, str(SRC))
else:
    # Por si ejecutas el script desde /notebooks o similar
    sys.path.insert(0, str((AQUI.parent / "src").resolve()))

from data_cleaner import DataCleanerConfig, DataCleaner, preprocess_data

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt


def crear_objetivo_volatilidad_futura(df: pd.DataFrame, velas_futuras: int = 12) -> pd.DataFrame:
    """
    Objetivo (y): volatilidad realizada en las próximas `velas_futuras` velas.
    - Usa std de log_return_close en una ventana de tamaño velas_futuras
    - Hace shift(-velas_futuras) para alinear el objetivo en t con el futuro (t+1 ... t+velas_futuras)
    """
    df = df.copy()
    df["vol_futura"] = (
        df["log_return_close"]
        .rolling(velas_futuras, min_periods=velas_futuras)
        .std()
        .shift(-velas_futuras)
    )
    return df


def preparar_X_y(df: pd.DataFrame, columna_y: str = "vol_futura"):
    """
    Selecciona features numéricas automáticamente (sin datetime/symbol/y).
    """
    excluir = {"datetime", "symbol", columna_y}
    columnas_X = [
        c for c in df.columns
        if c not in excluir and pd.api.types.is_numeric_dtype(df[c])
    ]

    X = df[columnas_X].copy()
    y = df[columna_y].copy()
    return X, y, columnas_X


def split_temporal(df: pd.DataFrame, test_ratio: float = 0.2):
    """
    Split simple temporal: train primero, test al final.
    """
    n = len(df)
    corte = int(n * (1 - test_ratio))
    train = df.iloc[:corte].copy()
    test = df.iloc[corte:].copy()
    return train, test


def main():
    # -------------------------
    # 1) Config: QQQ a 5 minutos
    # -------------------------
    cfg = DataCleanerConfig(
        source="alpaca",         
        symbol="QQQ",
        interval="5m",
        start_date="2022-01-01",
        end_date="2026-01-02",
    )

    # -------------------------
    # 2) Cargar + preprocesar
    # -------------------------
    cleaner = DataCleaner(cfg)
    df_raw = cleaner.cargar_datos()
    df = preprocess_data(df_raw)

    # -------------------------
    # 3) Crear objetivo: vol futura (próxima hora = 12 velas de 5m)
    # -------------------------
    VELAS_FUTURAS = 12  # 12*5m = 60 minutos
    df = crear_objetivo_volatilidad_futura(df, velas_futuras=VELAS_FUTURAS)

    # Quitamos NaNs del objetivo (al final) y por si quedó algo raro
    df = df.dropna(subset=["vol_futura"]).reset_index(drop=True)

    # 4) Split temporal

    train_df, test_df = split_temporal(df, test_ratio=0.2)

    X_train, y_train, cols_X = preparar_X_y(train_df, "vol_futura")
    X_test, y_test, _ = preparar_X_y(test_df, "vol_futura")

  
    modelo = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0, random_state=42))
    ])

    modelo.fit(X_train, y_train)
    pred_test = modelo.predict(X_test)

    mae = mean_absolute_error(y_test, pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred_test))

    r2 = r2_score(y_test, pred_test)

    print("\n=== Resultados (TEST) ===")
    print(f"MAE : {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"R2  : {r2:.4f}")
    print(f"Features usadas: {len(cols_X)}")

    salida = test_df[["datetime"]].copy()
    salida["vol_real"] = y_test.values
    salida["vol_pred"] = pred_test

    salida.to_csv("pred_vol_qqq_5m.csv", index=False)
    print("\n✅ Guardado: pred_vol_qqq_5m.csv")

    ultimos = salida.tail(600).copy()
    plt.figure()
    plt.plot(ultimos["datetime"], ultimos["vol_real"], label="Vol real")
    plt.plot(ultimos["datetime"], ultimos["vol_pred"], label="Vol pred")
    plt.xticks(rotation=45)
    plt.title("QQQ 5m - Volatilidad futura (próxima hora)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("vol_pred_vs_real.png", dpi=150)
    print("✅ Guardado: vol_pred_vs_real.png")

    # Info final: última predicción
    print("\nÚltima fila del test:")
    print(salida.tail(1).to_string(index=False))


if __name__ == "__main__":
    main()
