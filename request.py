import requests

# URL do endpoint de predição
url = 'http://127.0.0.1:5000/api/predict'

# Payload (dados da flor Iris para teste)
dados_amostra = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}

resposta = requests.post(url, json=dados_amostra)

print("Código HTTP de Status:", resposta.status_code)
print("Resposta JSON da API:", resposta.json())
