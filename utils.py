import io
import base64
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model

def plot_price_data(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df['timestamp'], df['close'], label="Preço de Fechamento")
    plt.xlabel("Data")
    plt.ylabel("Preço (USD)")
    plt.title("Histórico de Preços")
    plt.legend()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close()
    return image_base64

def garch_volatility_estimation(df):
    returns = np.log(df['close'] / df['close'].shift(1)).dropna()
    model = arch_model(returns, vol='Garch', p=1, q=1)
    model_fit = model.fit(disp="off")
    volatility_forecast = model_fit.forecast(horizon=5)
    return volatility_forecast.variance.values[-1, :][0]
