from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/test', methods=['POST'])
def test():
    return jsonify({"respuesta": "funciona"})

@app.route('/', methods=['GET'])
def home():
    return jsonify({"mensaje": "Bienvenido a la API Flask"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5005)
