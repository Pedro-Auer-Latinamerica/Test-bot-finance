let apiKey, secretKey;

function authenticate() {
  apiKey = document.getElementById("api-key").value;
  secretKey = document.getElementById("secret-key").value;

  fetch("/authenticate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ api_key: apiKey, secret_key: secretKey }),
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.status === "Autenticado com sucesso") {
        document.getElementById("auth-status").innerText = "Autenticado!";
        document.getElementById("auth-section").style.display = "none";
        document.getElementById("bot-section").style.display = "block";
      } else {
        document.getElementById("auth-status").innerText =
          "Falha na autenticação: " + data.error;
      }
    })
    .catch((error) => {
      document.getElementById("auth-status").innerText =
        "Erro de autenticação.";
    });
}

function plotPriceData() {
  fetch("/plot", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ api_key: apiKey, secret_key: secretKey }),
  })
    .then((response) => response.json())
    .then((data) => {
      document.getElementById("chart").src =
        "data:image/png;base64," + data.chart;
    })
    .catch((error) => console.error("Erro ao gerar o gráfico:", error));
}

function predictPrice() {
  fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ api_key: apiKey, secret_key: secretKey }),
  })
    .then((response) => response.json())
    .then((data) => {
      document.getElementById(
        "result"
      ).innerText = `Preços previstos: ${data.predicted_prices.join(
        ", "
      )}\nAção recomendada: ${data.recommended_action}`;
    })
    .catch((error) => console.error("Erro ao prever preço:", error));
}

function logout() {
  apiKey = "";
  secretKey = "";
  document.getElementById("auth-section").style.display = "block";
  document.getElementById("bot-section").style.display = "none";
  document.getElementById("auth-status").innerText = "";
  document.getElementById("result").innerText = "";
  document.getElementById("chart").src = "";
}
