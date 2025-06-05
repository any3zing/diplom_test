from flask import Flask, request, send_from_directory
from flask_socketio import SocketIO
from PIL import Image
import io
import base64
import json
import os

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/upload_multipart", methods=["POST"])
def upload_multipart():
    try:
        if "image" not in request.files or "detections" not in request.form:
            return {"error": "Missing parts"}, 400

        image_file = request.files["image"]
        detections_json = request.form["detections"]
        detection = json.loads(detections_json)

        timestamp = detection.get("timestamp", "")

        image_bytes = image_file.read()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        # Отправляем объект (не массив)
        socketio.emit("image", {
            "image": base64_image,
            "detection": detection,
            "timestamp": timestamp
        })

        return {"status": "ok"}
    except Exception as e:
        return {"error": str(e)}, 500



@socketio.on("connect")
def handle_connect():
    print("Client connected")

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=3000)
