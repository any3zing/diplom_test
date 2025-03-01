from flask import Flask, request, send_from_directory
from flask_socketio import SocketIO
import base64
import os
import json  # Импортируем библиотеку для работы с JSON

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/upload", methods=["POST"])
def upload():
    try:
        # Читаем JSON-данные из запроса
        data = request.json

        image = data.get("image")
        detections = data.get("detections", [])

        if not image:
            return {"error": "No image data"}, 400

        # Рассылаем изображение и данные о детектах клиентам
        socketio.emit("image", {"image": image, "detections": detections})
        return {"status": "ok"}
    except Exception as e:
        return {"error": str(e)}, 500

@socketio.on("connect")
def handle_connect():
    print("Client connected")

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=3000)