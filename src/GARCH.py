# src/GARCH.py
# GARCH(1,1) REAL OOS FINAL (H=2H) - versión ganadora:
# - Fit SOLO con TRAIN_FIT (parte inicial del train)
# - Sin calibración (porque en tu caso empeora)
# - Estacionalidad intradía (solo TRAIN) + suavizado=5
# - Fit window dentro de TRAIN_FIT = 16000
# - mean=Zero, dist=normal
#
# Output:
#   pred_vol_qqq_5m_garch11_oos_2h_BEST.csv
#   vol_pred_vs_real_garch11_oos_2h_BEST.png

print(">>> EJECUTANDO: GARCH(1,1) OOS BEST (H=2H) <<<")

from pathlib import Path
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from arch import arch_model
except ImportError:
    raise ImportError("Instala arch:  pip install arch")

AQUI = Path(__file__).resolve().parent
PROYECTO = AQUI.parent
SRC = PROYECTO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_cleaner import DataCleanerConfig, DataCleaner, preprocess_data


def interval_a_minutos(interval: str) -> int:
    s = str(interval).strip().lower()
    if s.endswith("m"):
        return int(s[:-1])
    if s.endswith("h"):
        return int(s[:-1]) * 60
    if s.endswith("d"):
        return int(s[:-1]) * 1440
    return 5


def suavizar_ciclico(valores: np.ndarray, half_window: int) -> np.ndarray:
    v = np.asarray(valores, dtype=float)
    w = int(half_window)
    if w <= 0:
        return v
    pad = np.r_[v[-w:], v, v[:w]]
    sm = pd.Series(pad).rolling(window=2*w+1, center=True, min_periods=1).mean().values
    return sm[w:-w]


class Garch11OOSBest:
    def __init__(
        self,
        source="alpaca",
        symbol="QQQ",
        interval="5m",
        start_date="2022-01-01",
        end_date="2026-01-02",
        horizonte_horas=2,
        test_ratio=0.2,
        train_val_ratio=0.2,          # SOLO para definir TRAIN_FIT (train inicial)
        escala_returns=100.0,
        usar_horario_regular=True,
        usar_estacionalidad=True,
        season_smooth_half_window=5,
        fit_window_train_bars=16000,
        mean_mode="Zero",
        dist_mode="normal",
    ):
        self.source = source
        self.symbol = symbol
        self.interval = interval
        self.start_date = start_date
        self.end_date = end_date

        self.minutos = interval_a_minutos(interval)
        self.N = int((horizonte_horas * 60) / self.minutos)

        self.test_ratio = float(test_ratio)
        self.train_val_ratio = float(train_val_ratio)

        self.escala_returns = float(escala_returns)
        self.usar_horario_regular = bool(usar_horario_regular)
        self.usar_estacionalidad = bool(usar_estacionalidad)
        self.season_smooth_half_window = int(season_smooth_half_window)

        self.fit_window_train_bars = int(fit_window_train_bars) if fit_window_train_bars is not None else None
        self.mean_mode = mean_mode
        self.dist_mode = dist_mode

        self.df = None
        self.corte_test = None
        self.corte_val = None

        self.season_arr = None
        self.params = None
        self.h_all = None
        self.r_scaled_all = None

    def cargar(self):
        cfg = DataCleanerConfig(
            source=self.source, symbol=self.symbol, interval=self.interval,
            start_date=self.start_date, end_date=self.end_date
        )
        df_raw = DataCleaner(cfg).cargar_datos()
        df = preprocess_data(df_raw)

        if self.usar_horario_regular:
            if df["datetime"].dt.tz is None:
                df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
            dt_ny = df["datetime"].dt.tz_convert("America/New_York")
            hora = dt_ny.dt.time
            ini = pd.to_datetime("09:30").time()
            fin = pd.to_datetime("16:00").time()
            df = df[(hora >= ini) & (hora <= fin)].copy().reset_index(drop=True)

        self.df = df
        return df

    def crear_target(self):
        df = self.df.copy()
        df["vol_futura_real"] = df["log_return_close"].rolling(self.N, min_periods=self.N).std().shift(-self.N)
        df = df.dropna(subset=["vol_futura_real"]).reset_index(drop=True)
        self.df = df
        return df

    def split(self):
        n = len(self.df)
        self.corte_test = int(n * (1 - self.test_ratio))

        train = self.df.iloc[:self.corte_test].copy()
        test = self.df.iloc[self.corte_test:].copy()

        n_train = len(train)
        self.corte_val = int(n_train * (1 - self.train_val_ratio))  # define TRAIN_FIT como [0:corte_val)
        train_fit = train.iloc[:self.corte_val].copy()

        return train_fit, test

    def ajustar_estacionalidad_train(self):
        if not self.usar_estacionalidad:
            self.season_arr = None
            return

        df = self.df
        corte = self.corte_test  # TODO el TRAIN

        if df["datetime"].dt.tz is None:
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

        dt_ny_all = df["datetime"].dt.tz_convert("America/New_York")
        mod_all = (dt_ny_all.dt.hour * 60 + dt_ny_all.dt.minute).astype(int).values

        df_train = df.iloc[:corte]
        dt_ny_train = df_train["datetime"].dt.tz_convert("America/New_York")
        mod_train = (dt_ny_train.dt.hour * 60 + dt_ny_train.dt.minute).astype(int).values

        absr = np.abs(df_train["log_return_close"].values.astype(float))
        med = float(np.median(absr))

        tmp = pd.DataFrame({"mod": mod_train, "absr": absr})
        season_map = tmp.groupby("mod")["absr"].mean().reindex(range(1440), fill_value=med).replace(0.0, med)
        season_map = pd.Series(
            suavizar_ciclico(season_map.values, self.season_smooth_half_window),
            index=season_map.index
        ).replace(0.0, med)

        season_arr = season_map.iloc[mod_all].values.astype(float)
        season_arr = np.where(np.isfinite(season_arr) & (season_arr > 0), season_arr, med)
        self.season_arr = season_arr

    def returns_modelo_all(self):
        r = self.df["log_return_close"].values.astype(float)
        if self.usar_estacionalidad:
            return r / self.season_arr
        return r

    def fit_train_fit(self):
        train_fit, _ = self.split()

        r_model_all = self.returns_modelo_all()
        self.r_scaled_all = (r_model_all * self.escala_returns).astype(float)

        r_train_fit_scaled = self.r_scaled_all[:self.corte_val]
        if self.fit_window_train_bars is not None:
            r_fit = r_train_fit_scaled[-self.fit_window_train_bars:]
        else:
            r_fit = r_train_fit_scaled

        warnings.filterwarnings("ignore")
        am = arch_model(r_fit, mean=self.mean_mode, vol="GARCH", p=1, q=1, dist=self.dist_mode, rescale=False)
        res = am.fit(disp="off", options={"maxiter": 4000})

        omega = float(res.params.get("omega"))
        alpha = float(res.params.get("alpha[1]"))
        beta = float(res.params.get("beta[1]"))
        mu = float(res.params.get("mu", 0.0))
        self.params = (omega, alpha, beta, mu)

        print("convergence_flag:", getattr(res, "convergence_flag", "NA"))
        try:
            print("opt_message:", res.optimization_result.message)
        except Exception:
            pass

        print("\n=== Params GARCH(1,1) (TRAIN_FIT) ===")
        print(f"omega={omega:.6f} alpha={alpha:.6f} beta={beta:.6f} alpha+beta={alpha+beta:.6f} mu={mu:.6f}")

    def filtrar_h_all(self):
        omega, alpha, beta, mu = self.params
        r = self.r_scaled_all
        h = np.empty(len(r), dtype=float)

        if (alpha + beta) < 0.999:
            h0 = omega / (1 - alpha - beta)
        else:
            h0 = float(np.var(r))

        h[0] = h0
        for t in range(1, len(r)):
            eps_prev = r[t - 1] - mu
            h[t] = omega + alpha * (eps_prev ** 2) + beta * h[t - 1]

        self.h_all = h

    # =========================
    # NUEVA FUNCIÓN (lo que pides)
    # =========================
    def predecir_vol_futura_en_t(self, t: int, devolver_path: bool = True):
        """
        Devuelve la volatilidad futura predicha desde el instante 't' para el horizonte del modelo (self.N velas),
        pensado para alimentar tu clasificación lateral/no-lateral a ~2h.

        - sigma_2h: escalar (vol 2h predicha), consistente con predecir_test()
        - sigma_path: vector de longitud N con sigma por vela futura (t+1..t+N) (opcional)

        Requisitos previos:
          cargar() -> crear_target() -> split() -> ajustar_estacionalidad_train()
          fit_train_fit() -> filtrar_h_all()
        """
        if self.params is None or self.h_all is None or self.r_scaled_all is None:
            raise RuntimeError(
                "Faltan estados internos. Llama antes a: ajustar_estacionalidad_train(), fit_train_fit(), filtrar_h_all()."
            )
        if self.usar_estacionalidad and self.season_arr is None:
            raise RuntimeError("usar_estacionalidad=True pero season_arr es None. Llama a ajustar_estacionalidad_train().")

        n = len(self.df)
        if t < 0 or t >= n:
            raise IndexError(f"t fuera de rango: {t} (n={n})")
        if (t + 1 + self.N) > n:
            return None  # no hay suficientes velas futuras

        omega, alpha, beta, mu = self.params
        ab = alpha + beta

        # h_{t+1} usando información hasta t (causal)
        eps_t = self.r_scaled_all[t] - mu
        h1 = omega + alpha * (eps_t ** 2) + beta * self.h_all[t]

        # path h_{t+1}..h_{t+N}
        hf = np.empty(self.N, dtype=float)
        hf[0] = h1
        for k in range(1, self.N):
            hf[k] = omega + ab * hf[k - 1]

        # sigma del modelo (sin estacionalidad) por paso
        sigma_model_path = np.sqrt(hf) / self.escala_returns

        # Escalar sigma_2h consistente con tu predecir_test(): sqrt(mean(hf))/scale * mean(season_next)
        vol_model_2h = np.sqrt(np.mean(hf)) / self.escala_returns

        if self.usar_estacionalidad:
            s_next = self.season_arr[t + 1: t + 1 + self.N]
            sigma_path = sigma_model_path * s_next
            sigma_2h = vol_model_2h * float(np.mean(s_next))
        else:
            sigma_path = sigma_model_path
            sigma_2h = vol_model_2h

        out = {
            "t": int(t),
            "datetime": self.df["datetime"].iloc[t],
            "sigma_2h": float(sigma_2h),
        }
        if devolver_path:
            out["sigma_path"] = sigma_path  # np.ndarray (N,)
        return out

    def predecir_test(self):
        _, test = self.split()
        omega, alpha, beta, mu = self.params
        ab = alpha + beta

        vol_real = test["vol_futura_real"].values.astype(float)
        vol_pred = np.full(len(test), np.nan, dtype=float)

        for i in range(len(test)):
            t = self.corte_test + i
            eps_t = self.r_scaled_all[t] - mu
            h1 = omega + alpha * (eps_t ** 2) + beta * self.h_all[t]

            hf = np.empty(self.N, dtype=float)
            hf[0] = h1
            for k in range(1, self.N):
                hf[k] = omega + ab * hf[k - 1]

            vol_model = np.sqrt(np.mean(hf)) / self.escala_returns
            if self.usar_estacionalidad:
                s_next = self.season_arr[t + 1: t + 1 + self.N]
                vol_pred[i] = vol_model * float(np.mean(s_next))
            else:
                vol_pred[i] = vol_model

        out = pd.DataFrame({
            "datetime": test["datetime"].values,
            "vol_real": vol_real,
            "vol_pred": vol_pred
        }).dropna().reset_index(drop=True)

        return out

    @staticmethod
    def metricas(out):
        y = out["vol_real"].values
        yhat = out["vol_pred"].values
        return (
            mean_absolute_error(y, yhat),
            np.sqrt(mean_squared_error(y, yhat)),
            r2_score(y, yhat),
        )

    def guardar(self, out):
        out.to_csv("pred_vol_qqq_5m_garch11_oos_2h_BEST.csv", index=False)
        print("\n✅ Guardado: pred_vol_qqq_5m_garch11_oos_2h_BEST.csv")

        ult = out.tail(600)
        plt.figure()
        plt.plot(ult["datetime"], ult["vol_real"], label="Vol real")
        plt.plot(ult["datetime"], ult["vol_pred"], label="Vol pred")
        plt.xticks(rotation=45)
        plt.title("QQQ 5m - GARCH(1,1) OOS vol futura (2h) BEST")
        plt.legend()
        plt.tight_layout()
        plt.savefig("vol_pred_vs_real_garch11_oos_2h_BEST.png", dpi=150)
        print("✅ Guardado: vol_pred_vs_real_garch11_oos_2h_BEST.png")

        print("\nÚltima fila del test:")
        print(out.tail(1).to_string(index=False))


def main():
    m = Garch11OOSBest(
        source="alpaca",
        symbol="QQQ",
        interval="5m",
        start_date="2022-01-01",
        end_date="2026-01-02",
        horizonte_horas=2,
        test_ratio=0.2,
        train_val_ratio=0.2,
        escala_returns=100.0,
        usar_horario_regular=True,
        usar_estacionalidad=True,
        season_smooth_half_window=5,
        fit_window_train_bars=16000,
        mean_mode="Zero",
        dist_mode="normal",
    )

    m.cargar()
    m.crear_target()

    # define splits
    m.split()

    # season con TODO TRAIN
    m.ajustar_estacionalidad_train()

    # fit solo train_fit
    m.fit_train_fit()

    # filtrar h
    m.filtrar_h_all()

    # (DEMO) ejemplo: forecast desde un instante t
    fc = m.predecir_vol_futura_en_t(t=10_000, devolver_path=True)
    if fc is not None:
        print("\n=== DEMO forecast en t=10000 ===")
        print("datetime:", fc["datetime"])
        print("sigma_2h:", fc["sigma_2h"])
        print("sigma_path (primeros 5):", np.round(fc["sigma_path"][:5], 8))

    # pred test
    out = m.predecir_test()
    mae, rmse, r2 = m.metricas(out)

    print(f"\n=== TEST OOS BEST | H={m.N} velas (~2h) ===")
    print(f"MAE : {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"R2  : {r2:.4f}")

    m.guardar(out)


if __name__ == "__main__":
    main()
