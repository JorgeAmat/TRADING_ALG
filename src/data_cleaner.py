import yfinance as yf
import pandas as pd
import numpy as np
import os
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import pandas as pd
from datetime import datetime
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


API_KEY = "PKCB2FGQTEQAIQXV5CXL6RJU4D" 
SECRET_KEY = "8gsG7rUpBvUaxAzw7HPw6EEypH42UpkR4VRYHaNJoYYU"
#Con esta clase definimos el objeto de configuración
class DataCleanerConfig:
    def __init__(self, source="alpaca", symbol="QQQ", interval="15m", start_date="2025-10-01", end_date="2025-10-22", csv_path=None):
        self.source = source
        self.symbol = symbol
        self.interval = interval
        self.start_date = start_date
        self.end_date = end_date
        self.csv_path = csv_path
      
        # Claves Alpaca
        self.api_key = API_KEY
        self.secret_key = SECRET_KEY
#DEFINIMOS EL CONSTRUCTOR
class DataCleaner:
    def __init__(self, cfg):
            self.cfg = cfg #Esta es la configuración, indica origen, fechas, rutas del csv...
            self.df = None

    # Cargamos el CSV    
        
    def _load_csv(self, path):
            if not os.path.exists(path):
                raise FileNotFoundError(f"❌ CSV no encontrado en: {path}")

            df = pd.read_csv(path)

            # Normalizamos columnas a minúsculas
            df.columns = [c.lower() for c in df.columns]

            # Aseguramos que datetime existe por si  acaso

            if "datetime" not in df.columns:
                raise ValueError("❌ El CSV debe contener una columna 'datetime'.")

            return df

    def _fetch_yfinance(self):

        
        #Descarga datos usando yfinance según la configuración dada en DataCleanerConfig
        import warnings
        warnings.filterwarnings("ignore")  # silencia los putos warnings

        data = yf.download(
            tickers=self.cfg.symbol,
            start=self.cfg.start_date,
            end=self.cfg.end_date,
            interval=self.cfg.interval,
            progress=False
        )

        if data is None or len(data) == 0:
            raise ValueError("yfinance no devolvió datos (posible problema de intervalo o fechas).")
        

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(col).lower() for col in data.columns]
        data.columns.name = None

        #peequeño test
        # print("✅ Data descargada:", len(data))
        # print(data)
        # Renombramos las columnas
        data = data.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        })

        #Aseguramos que la columna se llame datetime, para que no haaya confusiones
        data = data.reset_index()
        if "Date" in data.columns:
            data = data.rename(columns={"Date": "datetime"})
        elif "Datetime" in data.columns:
            data = data.rename(columns={"Datetime": "datetime"})
        else:
            data["datetime"] = data.index
        data.columns = [c.lower() for c in data.columns]
        return data         

    def _fetch_alpaca(self):
        """Descarga datos usando Alpaca según config."""
        if not self.cfg.api_key or not self.cfg.secret_key:
            raise ValueError("Debes pasar api_key y secret_key en DataCleanerConfig para usar Alpaca.")

        client = StockHistoricalDataClient(api_key=self.cfg.api_key, secret_key=self.cfg.secret_key)


        # Convertir interval "5m" → TimeFrame(5, Minute)
        interval_str = self.cfg.interval
        if interval_str.endswith("m"):
            minutes = int(interval_str.replace("m",""))
            timeframe = TimeFrame(amount=minutes, unit=TimeFrameUnit.Minute)
        else:
            raise ValueError(f"Intervalo no soportado para Alpaca: {interval_str}")
        
        symbols = self.cfg.symbol if isinstance(self.cfg.symbol, list) else [self.cfg.symbol]

        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=timeframe,
            start=datetime.fromisoformat(self.cfg.start_date),
            end=datetime.fromisoformat(self.cfg.end_date)
        )

        bars = client.get_stock_bars(request)
        df = bars.df.reset_index()

        # Normalización igual que en YFinance
        df = df.rename(columns={"timestamp": "datetime"})
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df.columns = [c.lower() for c in df.columns]
        df = df.sort_values("datetime").reset_index(drop=True)
        
        if "symbol" in df.columns:
            keep = [c for c in ["datetime","symbol","open","high","low","close","volume","vwap","trade_count"] if c in df.columns]
            df = df[keep]

            value_cols = [c for c in df.columns if c not in ["datetime","symbol"]]
            df = df.pivot(index="datetime", columns="symbol", values=value_cols)

            df.columns = [f"{a}_{b}".lower() for a, b in df.columns]
            df = df.reset_index()


        return df

    

    def cargar_datos(self):
        """
        Carga los datos desde la fuente configurada.
        Si la fuente es un CSV, lo lee desde la ruta indicada.
        Si la fuente es yfinance, los descarga.
        Guarda el resultado en self.df y lo devuelve.
        """
        if self.cfg.source == "csv":
            if not self.cfg.csv_path:
                raise ValueError("Debes indicar la ruta del CSV en cfg.csv_path")

            self.df = self._load_csv(self.cfg.csv_path)

        elif self.cfg.source == "yfinance":
            self.df = self._fetch_yfinance()

        elif self.cfg.source == "alpaca":
            self.df = self._fetch_alpaca()

        else:
            raise ValueError("Fuente de datos no válida. Usa 'csv', 'alpaca o 'yfinance'.")

        return self.df
    

def preprocess_data(df, target_symbol):
    """
    Preprocesado causal (sin mirar futuro).
    - Ordena por datetime
    - Limpia volume (0 -> NaN) y rellena SOLO con pasado (ffill)
    - Crea features de retornos + volatilidad + momentum básico
    """

    df = df.copy()

    # 1) datetime + orden
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    
        # --- BLOQUE NUEVO (mínimo) ---
    # Normaliza target_symbol a como viene en columnas (en tu caso: qqq, tlt, etc.)
    ts = str(target_symbol).lower()

    # Si el df viene en formato ancho (close_qqq, open_qqq...), copiamos esas columnas a nombres genéricos
    mapping = {
        "open": f"open_{ts}",
        "high": f"high_{ts}",
        "low":  f"low_{ts}",
        "close": f"close_{ts}",
        "volume": f"volume_{ts}",
        "vwap": f"vwap_{ts}",
        "trade_count": f"trade_count_{ts}",
    }

    # Asegura que existe el close del activo objetivo
    if mapping["close"] in df.columns:
        for base, col in mapping.items():
            if col in df.columns:
                df[base] = df[col]
    # Si ya viene en formato "simple" (close), no hacemos nada
    elif "close" not in df.columns:
        raise ValueError(f"Falta columna `close` o `{mapping['close']}` para target_symbol={ts}")
    # --- FIN BLOQUE NUEVO ---


    eps = 1e-12
    W = 20  # ventana estándar para rolling

    # 2) asegurar numéricos + clips
    for c in ["open", "high", "low", "close", "volume", "vwap", "trade_count"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "close" not in df.columns:
        raise ValueError("Falta columna `close`.")

    df["close"] = df["close"].clip(lower=eps)

    # 3) volumen (sin leakage: NO interpolación lineal)
    if "volume" in df.columns:
        df.loc[df["volume"] == 0, "volume"] = np.nan
        df["volume"] = df["volume"].ffill()  # solo pasado
        vol_mean = df["volume"].rolling(W, min_periods=W).mean()
        vol_std  = df["volume"].rolling(W, min_periods=W).std()
        df["volume_norm"] = (df["volume"] - vol_mean) / (vol_std + 1e-9)

    # 4) trade_count (alpaca) opcional, también causal
    if "trade_count" in df.columns:
        df.loc[df["trade_count"] == 0, "trade_count"] = np.nan
        df["trade_count"] = df["trade_count"].ffill()
        tc_mean = df["trade_count"].rolling(W, min_periods=W).mean()
        tc_std  = df["trade_count"].rolling(W, min_periods=W).std()
        df["trade_count_norm"] = (df["trade_count"] - tc_mean) / (tc_std + 1e-9)

    # 5) returns y log-returns
    df["return"] = df["close"].pct_change()

    log_close = np.log(df["close"])
    df["log_return_close"] = log_close.diff()

    lr_mean = df["log_return_close"].rolling(W, min_periods=W).mean()
    lr_std  = df["log_return_close"].rolling(W, min_periods=W).std()
    df["log_return_norm"] = (df["log_return_close"] - lr_mean) / (lr_std + 1e-9)

    # 6) ratios OHLC en log (si existen)
    if "open" in df.columns:
        df["open"] = df["open"].clip(lower=eps)
        df["log_ret_oc"] = np.log(df["close"] / df["open"])

    if "high" in df.columns and "low" in df.columns:
        df["high"] = df["high"].clip(lower=eps)
        df["low"]  = df["low"].clip(lower=eps)
        df["log_ret_hl"] = np.log(df["high"] / df["low"])

    if "high" in df.columns:
        df["log_ret_ch"] = np.log(df["close"] / df["high"])

    if "low" in df.columns:
        df["log_ret_cl"] = np.log(df["close"] / df["low"])

    # 7) VWAP (alpaca) opcional
    if "vwap" in df.columns:
        df["vwap"] = df["vwap"].clip(lower=eps)
        df["log_ret_vwap"] = np.log(df["vwap"]).diff()

    # 8) lags de log_return_close (momentum simple)
    for lag in [1, 3, 5]:
        df[f"log_return_lag{lag}"] = df["log_return_close"].shift(lag)

    # 9) volatilidad rolling
    df["vol_rolling"] = df["log_return_close"].rolling(W, min_periods=W).std()

    # 10) RSI (14) simple y causal
    rsi_n = 14
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(rsi_n, min_periods=rsi_n).mean()
    loss = (-delta.clip(upper=0)).rolling(rsi_n, min_periods=rsi_n).mean()
    rs = gain / (loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    # 11) MACD (en log-precio)
    ema12 = log_close.ewm(span=12, adjust=False).mean()
    ema26 = log_close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # 12) limpiar NaNs iniciales por rolling/shift
    df = df.dropna().reset_index(drop=True)
    return df


# TEST:
# cfg = DataCleanerConfig(source="yfinance",symbol="QQQ",interval="15m",start_date="2025-10-01",end_date="2025-10-22")
# symbol_df = DataCleaner(cfg)
# raw_df = symbol_df.cargar_datos()

# #Preprocesar
# df_limpio = preprocess_data(raw_df)

# print(df_limpio)
# print(f"\nFilas finales: {len(df_limpio)}")


# TEST: 

if __name__ == "__main__":


    # TEST CARGAR_DATOS
    # TLT --> ETF Bono americano, IEGA.L --> BNDX Bono internacional menos USA, VXX --> nivel de miedo de los inversores
    cfg = DataCleanerConfig(source="alpaca",symbol=["TLT", "VXX", "BNDX"],interval="15m",start_date="2025-10-20", end_date= "2025-10-21")
    #   TEST CARGAR_DATOS
    symbol_data = DataCleaner(cfg)
    datos = symbol_data.cargar_datos()

    print(datos)
    print(f"\nFilas descargadas: {len(datos)}")
   

# TEST PREPROCESS_DATA:
   

