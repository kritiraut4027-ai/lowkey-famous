import os
import sys
import uuid
from pathlib import Path
from flask import Flask, request, send_from_directory, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Robust path resolution anchored to project root
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

UPLOAD_FOLDER = BASE_DIR / "uploads"
DATASET_FOLDER = BASE_DIR / "dataset"
FRONTEND_FOLDER = BASE_DIR / "frontend"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

from backend.compare_faces import (
    find_hamshakal,
    NoFaceDetectedError,
    MultipleFacesDetectedError,
    InvalidImageError,
    FaceAnalysisError,
    ModelInitializationError,
    load_celebrity_embeddings,
    get_celebrity_image_map,
)

app = Flask(
    __name__,
    static_folder=str(FRONTEND_FOLDER),
    static_url_path="",
)

# Enable CORS for all routes (allows decoupled frontends on Vercel/Netlify)
CORS(app)

# 8 MB max upload limit
app.config["MAX_CONTENT_LENGTH"] = int(
    os.environ.get("MAX_UPLOAD_MB", 8)
) * 1024 * 1024

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}


def allowed_file(filename: str, content_type: str = "") -> bool:
    has_allowed_ext = (
        "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )
    if not has_allowed_ext:
        return False
    if content_type and content_type.lower() not in ALLOWED_MIME_TYPES:
        return False
    return True


@app.route("/")
def index():
    """Serves the frontend application if available in the frontend folder."""
    index_file = FRONTEND_FOLDER / "index.html"
    if index_file.exists():
        return send_from_directory(str(FRONTEND_FOLDER), "index.html")
    return jsonify({
        "status": "ok",
        "service": "Hamshakal Finder API",
        "version": "2.0.0",
        "docs": "POST /upload with multipart/form-data key 'image'"
    })


@app.route("/health", methods=["GET"])
def health():
    """Liveness probe for cloud deployments (Render, Railway, Hugging Face)."""
    try:
        embeddings = load_celebrity_embeddings()
        num_celebs = len(embeddings)
        return jsonify({
            "status": "healthy",
            "celebrities_indexed": num_celebs
        }), 200
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    """Serves temporary upload preview if retained."""
    safe_name = secure_filename(filename)
    return send_from_directory(str(UPLOAD_FOLDER), safe_name)


@app.route("/dataset/<path:filename>")
def serve_dataset_image(filename):
    """Serves celebrity reference images with path support and case-insensitive matching fallback."""
    target = (DATASET_FOLDER / filename).resolve()
    
    # Path traversal protection
    if DATASET_FOLDER.resolve() not in target.parents and target != DATASET_FOLDER.resolve():
        return jsonify({"error": "Access denied"}), 403

    if target.exists() and target.is_file():
        # Relative path from DATASET_FOLDER
        rel_path = target.relative_to(DATASET_FOLDER.resolve())
        return send_from_directory(str(DATASET_FOLDER), str(rel_path).replace("\\", "/"))

    # Linux case-sensitive fallback: check if lowercased match exists
    file_lower = Path(filename).name.lower()
    for existing in DATASET_FOLDER.rglob("*.*"):
        if existing.is_file() and existing.name.lower() == file_lower:
            rel_path = existing.relative_to(DATASET_FOLDER.resolve())
            return send_from_directory(str(DATASET_FOLDER), str(rel_path).replace("\\", "/"))

    return jsonify({"error": "Celebrity image not found"}), 404


@app.route("/upload", methods=["POST"])
def upload_image():
    """
    Main face match endpoint. Accepts multipart/form-data with key 'image'.
    Performs face detection, ArcFace embedding extraction, and cosine similarity comparison.
    Automatically removes temporary upload file to prevent storage exhaustion.
    """
    if "image" not in request.files:
        return jsonify({
            "status": "error",
            "error": "No image field found in upload request."
        }), 400

    image_file = request.files["image"]
    if not image_file or not image_file.filename:
        return jsonify({
            "status": "error",
            "error": "No file selected."
        }), 400

    content_type = image_file.content_type or ""
    if not allowed_file(image_file.filename, content_type):
        return jsonify({
            "status": "invalid_image",
            "error": "Unsupported file format. Please upload a JPG, PNG, or WEBP image."
        }), 400

    # Generate a unique secure temporary filename
    ext = image_file.filename.rsplit(".", 1)[1].lower() if "." in image_file.filename else "jpg"
    unique_filename = f"{uuid.uuid4().hex}.{ext}"
    temp_image_path = UPLOAD_FOLDER / unique_filename

    try:
        image_file.save(str(temp_image_path))

        # Perform face recognition and similarity search
        best_match_key, percentage, celeb_filename, display_name, raw_similarity = find_hamshakal(
            temp_image_path
        )

        return jsonify({
            "status": "ok",
            "celebrity_key": best_match_key,
            "celebrity_name": display_name,
            "celebrity_image_url": f"/dataset/{celeb_filename}",
            "user_image_url": f"/uploads/{unique_filename}",
            "match_percentage": percentage,
            "similarity_score": round(raw_similarity, 4),
            "score_type": "cosine_similarity"
        }), 200

    except NoFaceDetectedError as e:
        return jsonify({
            "status": "no_face",
            "error": str(e)
        }), 200

    except MultipleFacesDetectedError as e:
        return jsonify({
            "status": "multiple_faces",
            "error": str(e)
        }), 200

    except InvalidImageError as e:
        return jsonify({
            "status": "invalid_image",
            "error": str(e)
        }), 400

    except ModelInitializationError as e:
        return jsonify({
            "status": "error",
            "error": f"Model failed to initialize: {e}"
        }), 503

    except FaceAnalysisError as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

    except Exception as e:
        return jsonify({
            "status": "error",
            "error": f"Unexpected server error during face matching: {e}"
        }), 500

    finally:
        # Privacy & storage hygiene: remove temporary upload from disk
        # To retain for preview, set RETAIN_UPLOADS=true
        if os.environ.get("RETAIN_UPLOADS", "false").lower() != "true":
            if temp_image_path.exists():
                try:
                    os.remove(temp_image_path)
                except OSError:
                    pass


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    print(f"[Hamshakal Finder] Starting server on {host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug)
