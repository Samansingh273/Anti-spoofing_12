from flask import Flask, render_template, jsonify, send_from_directory
from deepface import DeepFace
import os
import subprocess

app = Flask(__name__)
CAPTURE_DIR = "captures"
FACE_DB_DIR = "face_db"
main_process = None

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/start')
def start():
    global main_process
    if main_process is None or main_process.poll() is not None:
        main_process = subprocess.Popen(["python", "main.py"])
        return jsonify({"status": "Started main.py"})
    return jsonify({"status": "main.py already running"})

@app.route('/stop')
def stop():
    global main_process
    if main_process and main_process.poll() is None:
        main_process.kill()
        main_process.wait()
        main_process = None
        return jsonify({"status": "Forcefully stopped main.py"})
    return jsonify({"status": "main.py is not running"})

def match_face(image_path):
    try:
        result = DeepFace.find(
            img_path=image_path,
            db_path=FACE_DB_DIR,
            model_name='ArcFace',
            detector_backend='opencv',
            enforce_detection=False,
            silent=True
        )
        if result and not result[0].empty:
            match_file = result[0].iloc[0]['identity']
            return os.path.splitext(os.path.basename(match_file))[0]
        return "Unknown"
    except Exception as e:
        print(f"[ERROR] Matching {image_path}: {e}")
        return "Error"

@app.route('/captures')
def list_captures():
    files = sorted(os.listdir(CAPTURE_DIR), reverse=True)
    data = []
    for file in files:
        filepath = os.path.join(CAPTURE_DIR, file)
        match = match_face(filepath)
        data.append({"filename": file, "match": match})
    return jsonify(data)

@app.route('/captures/<filename>')
def get_image(filename):
    return send_from_directory(CAPTURE_DIR, filename)

if __name__ == '__main__':
    app.run(debug=True)
