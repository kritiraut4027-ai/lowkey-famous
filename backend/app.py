from flask import Flask, request, send_from_directory, jsonify
from flask_cors import CORS
import os
import sys
from pathlib import Path 
#from scripts.compare_faces import find_hamshakal
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_FOLDER=os.path.join(BASE_DIR,"uploads")
DATASET_FOLDER= os.path.join(BASE_DIR,"dataset")

app = Flask(__name__)
CORS(app)

os.makedirs(UPLOAD_FOLDER,exist_ok=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.compare_faces import find_hamshakal



@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER,filename)

@app.route("/dataset/<filename>")
def serve_image(filename):
    return send_from_directory(DATASET_FOLDER,filename)


@app.route("/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return "No image uploaded."
    
    image = request.files["image"]
    filename = image.filename

    image_path= os.path.join(UPLOAD_FOLDER,image.filename)
    image.save(image_path)


    best_match, percentage,celeb_image_path = find_hamshakal(image_path)

    return jsonify({
    "status": "ok",
    "celebrity_name": str(best_match),
    "celebrity_image_url": f"/dataset/{best_match}.jpg",
    "user_image_url": f"/uploads/{filename}",
    "match_percentage": float(percentage)
})
    
   
    
if __name__ == "__main__":app.run(debug=True)
