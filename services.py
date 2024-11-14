import pandas as pd
import ccxt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch import optim
from models import LSTMModel
from utils import garch_volatility_estimation

def connect_to_binance(api_key, secret_key):
    """Conectar à Binance usando as chaves fornecidas."""
    return ccxt.binance({
        'apiKey': api_key,
        'secret': secret_key,
        'enableRateLimit': True,
    })

def fetch_ohlcv(exchange, symbol='BTC/USDT', timeframe='1h', limit=500):
    """Buscar dados históricos de preços da Binance."""
    data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def train_pytorch_model(df, epochs=10, learning_rate=0.001):
    """Treina uma rede neural LSTM para previsão de preços usando PyTorch."""
    if len(df) < 11:
        raise ValueError("O DataFrame precisa ter pelo menos 11 registros para treinamento.")

    # Preprocessamento dos dados
    df_copy = df.copy()
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_copy['scaled_close'] = scaler.fit_transform(df_copy[['close']])

    X, y = [], []
    look_back = 10
    for i in range(look_back, len(df_copy)):
        X.append(df_copy['scaled_close'].values[i-look_back:i])
        y.append(df_copy['scaled_close'].values[i])

    X, y = np.array(X), np.array(y)
    X = torch.from_numpy(X).float().unsqueeze(2)  # Adiciona dimensão para PyTorch (batch_size, seq_length, input_size)
    y = torch.from_numpy(y).float()

    # Inicializando o modelo, critério de perda e otimizador
    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Treinamento
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred.squeeze(), y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    return model, scaler

def predict_price_with_model(model, scaler, df):
    """Usa o modelo treinado para fazer previsões de preços futuros."""
    if len(df) < 10:
        raise ValueError("O DataFrame precisa ter pelo menos 10 registros para fazer previsões.")

    # Preprocessamento para previsão
    df_copy = df.copy()
    scaled_data = scaler.transform(df_copy[['close']])

    X_test = []
    look_back = 10
    for i in range(look_back, len(scaled_data)):
        X_test.append(scaled_data[i-look_back:i, 0])

    X_test = np.array(X_test)
    X_test = torch.from_numpy(X_test).float().unsqueeze(2)  # Adiciona dimensão para PyTorch

    model.eval()
    with torch.no_grad():
        predictions = model(X_test).squeeze().numpy()  # Realiza as previsões e converte para numpy
    predicted_prices = scaler.inverse_transform(predictions.reshape(-1, 1))

    return predicted_prices.flatten()

def decision_theory_strategy(df, model, scaler):
    """Estratégia de decisão baseada na volatilidade e previsão de preços."""
    volatility = garch_volatility_estimation(df)
    last_price = df['close'].iloc[-1]
    predicted_prices = predict_price_with_model(model, scaler, df)
    if len(predicted_prices) == 0:
        raise ValueError("Nenhuma previsão disponível do modelo.")

    predicted_price = predicted_prices[-1]

    if predicted_price > last_price and volatility < 0.05:
        return 'buy'
    elif predicted_price < last_price and volatility > 0.05:
        return 'sell'
    else:
        return 'hold'
