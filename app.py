import os
from flask import Flask, jsonify, request
import pandas as pd
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.svm import SVC
from flask_cors import CORS

# Cargar archivo CSV
def cargar_datos(ruta_csv):
    datos = pd.read_csv(ruta_csv)
    return datos['pregunta'], datos['pregunta_normalizada'], datos['intención'], datos['categoria'], datos['respuesta']

def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'[^\w\s]', '', texto)  # Eliminar puntuación y espacios en blanco extra
    texto = texto.strip()
    return texto

# Entrenar Modelo usando SVC
def entrenar_modelo(preguntas_normalizadas, intenciones, categorias, respuestas):
    # Concatenar las preguntas normalizadas con las categorías para incluir más contexto en el entrenamiento
    preguntas_con_categoria = [f"{pregunta} {categoria}" for pregunta, categoria in zip(preguntas_normalizadas, categorias)]
    stop_words = stopwords.words('spanish')
    
    # Dividir los datos en conjunto de entrenamiento y prueba (80% entrenamiento, 20% prueba)
    #X_train, X_test, y_train, y_test = train_test_split(preguntas_con_categoria, intenciones, test_size=0.2, random_state=42)

    # Crear el pipeline con TfidfVectorizer y SVC
    modelo = make_pipeline(
        TfidfVectorizer(stop_words=stop_words),
        SVC(probability=True)  # SVM con probabilidad
    )

    # Entrenar el modelo usando el conjunto de entrenamiento
    #modelo.fit(X_train, y_train)
    
    # Hacer predicciones en el conjunto de prueba
    #y_pred = modelo.predict(X_test)
    
    # Calcular el accuracy
    #accuracy = accuracy_score(y_test, y_pred)

    # Mostrar el accuracy y otros informes
    #print(f"Accuracy del modelo: {accuracy:.2f}")
    #print("Reporte de clasificación:\n", classification_report(y_test, y_pred))
    #print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))
    
    #return modelo

    # Entrenar el modelo usando las preguntas con categoría y la intención como etiquetas
    modelo.fit(preguntas_con_categoria, intenciones)
    return modelo

# Saltos de línea en respuestas
def formatear_respuesta(texto):
    return texto.replace('\n', '<br>')


# Detecta URLs en una cadena de texto y las convierte en enlaces clickeables
def hacer_urls_clickeables(texto):
    url_regex = r'(https?://[^\s]+)'
    
    # Reemplaza cada URL por una etiqueta <a> HTML
    texto_con_enlaces = re.sub(url_regex, r'<a href="\1" target="_blank">\1</a>', texto)
    return texto_con_enlaces

# Predecir respuesta
def predecir_respuesta(modelo, pregunta, datos, umbral=0.05):
    proba = modelo.predict_proba([pregunta])
    mejor_prediccion = proba.max()

    enlace_ticket = '\n\nSi este mensaje no responde tu pregunta o deseas continuar hablando con un agente, puedes <a href="https://clientes.koryfi.com/submitticket.php?step=2&deptid=1" target="_blank">abrir un ticket de soporte</a>.'

    if mejor_prediccion > umbral:
        # Obtener la intención predicha y la respuesta correspondiente
        intencion_predicha = modelo.predict([pregunta])[0]
        respuesta = datos.loc[datos['intención'] == intencion_predicha, 'respuesta'].values[0]
        
        # Aplica formato de URLs y saltos de línea solo a la respuesta del CSV de la empresa
        respuesta_formateada = formatear_respuesta(respuesta)
        enlace_ticket_formateado = formatear_respuesta(enlace_ticket)
        respuesta_con_enlaces = hacer_urls_clickeables(respuesta_formateada)
        return respuesta_con_enlaces + " " + enlace_ticket_formateado
    else:
        # Respuesta cuando no se encuentra una intención con suficiente confianza
        return 'No tengo la respuesta a esta pregunta, puedes intentar <a href="https://clientes.koryfi.com/submitticket.php?step=2&deptid=1" target="_blank">abrir un ticket aquí</a>.'


class Chatbot:
    def __init__(self, ruta_csv):
        # Carga, procesa los datos, entrena el modelo, guarda el csv para mapear las respuestas
        self.preguntas, self.preguntas_normalizadas, self.intenciones, self.categorias, self.respuestas = cargar_datos(ruta_csv)
        self.modelo = entrenar_modelo(self.preguntas_normalizadas, self.intenciones, self.categorias, self.respuestas)
        self.datos_csv = pd.read_csv(ruta_csv)
    
    def responder(self, pregunta):
        # Usar la función predecir_respuesta para obtener la respuesta
        respuesta = predecir_respuesta(self.modelo, pregunta, self.datos_csv)
        return respuesta

app = Flask(__name__)
CORS(app)

# Inicializar el chatbot globalmente
bot = Chatbot('./faq12.csv')

@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    # Recibe la pregunta en formato JSON desde el frontend
    data = request.get_json()
    pregunta = data.get('pregunta')
    
    if not pregunta:
        return jsonify({"respuesta": "Por favor, ingresa una pregunta válida."}), 400
    
    # Obtiene la respuesta del chatbot
    respuesta = bot.responder(pregunta)
    
    # Envia la respuesta de vuelta al frontend
    return jsonify({"respuesta": respuesta})

@app.route('/', methods=['GET'])
def home():
    return jsonify({"mensaje": "Bienvenido a la API Flask"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
