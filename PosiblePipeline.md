### PIPELINE IA

Consistiría en hacer varios modelos y juntarlos todos al final:
- Modelo de tendencia --> NBEATS
- Modelo de volatilidad --> GARCH o ML
- Modelo clasificador --> LSTM (usando features, prediccion de tendencia y predicción de volatilidad)

Dicen tambien de usar probability thresholding --> ayuda mucho da probabilidades de que el precio suba, baje o se quede igual. 

Nuestro pipeline sería entonces el siguiente:

Market Data
    ↓
Feature Engineering
    ↓
Trend Model (NBEATS)
    ↓
Volatility Model
    ↓
Classifier
    ↓
Probability filter (opcional)
    ↓
Trading Strategy