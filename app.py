from flask import Flask, request, jsonify, render_template
from services import connect_to_binance, fetch_ohlcv, train_pytorch_model, predict_price_with_model, decision_theory_strategy
from utils import plot_price_data

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/authenticate', methods=['POST'])
def authenticate():
    data = request.get_json()
    api_key = data.get('api_key')
    secret_key = data.get('secret_key')
    try:
        exchange = connect_to_binance(api_key, secret_key)
        exchange.fetch_balance()  # Testa conexão
        return jsonify({"status": "Autenticado com sucesso"}), 200
    except Exception as e:
        return jsonify({"status": "Falha na autenticação", "error": str(e)}), 401

@app.route('/plot', methods=['POST'])
def plot_price_data_route():
    data = request.get_json()
    api_key = data.get('api_key')
    secret_key = data.get('secret_key')
    exchange = connect_to_binance(api_key, secret_key)
    df = fetch_ohlcv(exchange)
    chart_image = plot_price_data(df)
    return jsonify({"chart": chart_image})

@app.route('/predict', methods=['POST'])
def predict_price():
    data = request.get_json()
    api_key = data.get('api_key')
    secret_key = data.get('secret_key')
    exchange = connect_to_binance(api_key, secret_key)
    df = fetch_ohlcv(exchange)

    model, scaler = train_pytorch_model(df)
    predicted_prices = predict_price_with_model(model, scaler, df)
    action = decision_theory_strategy(df, model, scaler)

    return jsonify({
        "predicted_prices": predicted_prices.tolist(),
        "recommended_action": action
    })

if __name__ == "__main__":
    app.run(debug=True)
