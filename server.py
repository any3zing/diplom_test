from flask import Flask, request, send_from_directory
from flask_socketio import SocketIO
import base64
import os

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/upload", methods=["POST"])
def upload():
    data = request.json.get("image")
    if not data:
        return {"error": "No image data"}, 400

    socketio.emit("image", data)  # Рассылаем изображение клиентам
    return {"status": "ok"}

@socketio.on("connect")
def handle_connect():
    print("Client connected")

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=3000)
