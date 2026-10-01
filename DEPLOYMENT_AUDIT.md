# Deployment Audit Report: Hamshakal Finder MVP

**Date:** September 2026  
**Auditor:** Senior ML Engineer & Backend Deployment Architect  
**Project:** Hamshakal Finder (Celebrity Lookalike AI)

---

## 1. Executive Summary

Hamshakal Finder is a lightweight AI demonstration application that compares an uploaded user selfie with a gallery of 8 Bollywood celebrity facial embeddings using cosine similarity.

While the core machine learning logic functions locally on Windows, the application currently cannot be reliably deployed to free or low-resource cloud platforms (e.g., Render, Railway, Hugging Face Spaces, Vercel) due to:
1. **Critical Process Termination Bugs:** The backend calls Python's `exit()` directly within request handlers when image reading fails, when no face is found, or when multiple faces are detected, terminating the entire server process.
2. **Excessive Memory & Model Overhead:** `insightface.app.FaceAnalysis()` is initialized without restricting allowed modules, loading 5 ONNX models into memory (including a 137 MB 3D landmark mesh model and gender/age models) despite only face detection and recognition being used. This inflates memory consumption to 532 MB during inference (exceeding typical 512 MB free-tier limits) and slows CPU inference to ~6.9 seconds.
3. **Redundant Disk I/O Per Request:** The backend reads and parses all `.npy` files from disk on every single HTTP request instead of caching them in memory at startup.
4. **Hardcoded Localhost & Working Directory Assumptions:** The frontend hardcodes `http://127.0.0.1:5000` for both API endpoints and image asset URLs. The backend relies on relative paths like `"embedding"` and `"dataset"` without anchoring to the project root, failing whenever the current working directory shifts.
5. **Bloated & Incompatible Dependencies:** `requirements.txt` contains 37 packages (515+ MB installed), including unused heavy packages like `pandas` (~60 MB) and `opencv-python` with GUI/X11 dependencies that break headless Linux containers. Additionally, `requirements.txt` was encoded in UTF-16LE, causing standard package managers to choke.
6. **Path Traversal & Security Vulnerabilities:** Uploads are saved directly using the raw `image.filename` with no sanitation, no MIME/extension validation, no upload size caps (`MAX_CONTENT_LENGTH`), and no cleanup of uploaded user photos.
7. **Filesystem Case Sensitivity Bug:** Celebrity image serving assumes case-insensitive matching (`/dataset/{best_match}.jpg`). On Linux/Docker, `best_match="anushka"` fails because the file on disk is `Anushka.jpg`.

---

## 2. Existing Architecture Overview

```mermaid
flowchart TD
    User["User Browser"] -->|POST /upload (multipart/form-data)| Flask["Flask Backend (backend/app.py)"]
    Flask -->|Save raw upload| Uploads["uploads/ folder (unbounded storage)"]
    Flask -->|find_hamshakal(image_path)| Compare["backend/compare_faces.py"]
    Compare -->|Initialize on import| InsightFace["FaceAnalysis (buffalo_l)"]
    InsightFace -->|Loads 5 ONNX Models| Memory["532 MB RAM Footprint"]
    Compare -->|Every request: read all .npy| Disk["embedding/*.npy (8 files)"]
    Compare -->|Cosine Similarity| Math["np.dot / (norm1 * norm2)"]
    Compare -->|Return best match & %| Flask
    Flask -->|JSON Response| User
    User -->|GET http://127.0.0.1:5000/dataset/...| Flask
```

### Component Inventory

| Component | Path | Responsibility | Deployment Status |
| :--- | :--- | :--- | :--- |
| **Backend Entry** | `backend/app.py` | Flask API routes (`/upload`, `/uploads/<filename>`, `/dataset/<filename>`) | Critical blockers: `app.run(debug=True)` without host/port binding, unsafe file handling |
| **Inference Logic** | `backend/compare_faces.py` | Face detection, embedding extraction, similarity matching | Critical blockers: `exit()` calls kill server process; loads 5 models; un-anchored CWD paths; disk reads per request |
| **Embedding Prep** | `scripts/generating_embedding.py` | Pre-computes 512-d embeddings for `dataset/` | Hardcoded `ctx_id=0` (GPU warning on CPU); CWD path dependency |
| **Frontend UI** | `frontend/index (1).html`, `style.css`, `script.js` | Drag-and-drop upload, scanning animation, results card | Critical blockers: Misnamed HTML (`index (1).html`), hardcoded `http://127.0.0.1:5000` |
| **Celebrity Images** | `dataset/*.jpg` (8 files) | Reference images for lookalikes | 0.43 MB; Case-sensitivity bug (`Anushka.jpg` vs `anushka.npy`) |
| **Pre-computed Embeddings** | `embedding/*.npy` (8 files) | 512-dimensional float32 vectors | 17.4 KB total; 100% verified compatible with `buffalo_l` ArcFace |
| **User Uploads** | `uploads/` | Stored uploaded user photos | Unbounded disk leak (16.4 MB of test images currently stored and tracked) |

---

## 3. Dependency Analysis

### Installed Packages Size Breakdown (`site-packages` total: 515.8 MB)

| Package | Installed Size | Required by App? | Notes / Recommendation |
| :--- | :--- | :--- | :--- |
| `cv2` (`opencv-python`) | 112.4 MB | **Replace** | Full GUI package. Replace with `opencv-python-headless` (saves ~30-50 MB and prevents Linux X11/libGL missing library crashes). |
| `scipy` + `scipy.libs` | 127.6 MB | Retain (InsightFace dep) | InsightFace declares `scipy` as a required dependency. |
| `pandas` | 59.9 MB | **Remove** | Completely unused in the codebase. Safe to eliminate immediately. |
| `onnxruntime` | 41.6 MB | Retain | Required for ONNX model CPU inference. |
| `onnx` | 34.5 MB | Retain | Required by InsightFace model zoo. |
| `numpy` + `numpy.libs` | 50.7 MB | Retain | Core vector and array operations. |
| `skimage` (`scikit-image`) | 23.1 MB | Retain (InsightFace dep) | InsightFace uses face alignment transform utilities. |
| `PIL` (`pillow`) | 15.3 MB | **Remove** | App uses OpenCV for image decoding. Not needed in production. |
| `networkx` | 14.7 MB | Retain (skimage dep) | Transitive dependency of scikit-image. |
| `Flask` + `Werkzeug` | ~10 MB | Retain | Web framework and WSGI utilities. |
| `flask-cors` | < 1 MB | Retain | Handles cross-origin requests between frontend and backend. |
| `gunicorn` | < 1 MB | Retain | Production WSGI HTTP server for Linux containers. |
| `insightface` | 2.5 MB | Retain | Face analysis and ArcFace embeddings. |

---

## 4. Model Architecture & Resource Bottlenecks

### Current InsightFace Model Bundle (`buffalo_l`)

InsightFace stores models in `~/.insightface/models/buffalo_l/`. When `FaceAnalysis()` is initialized without restricting `allowed_modules`, it loads:

| Model File | Task Name | ONNX Size | Loaded by Default? | Required by Hamshakal Finder? | Action |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `det_10g.onnx` | Detection (`SCRFD`) | 16.1 MB | Yes | **YES** | Retain (detects face bounding box & 5 keypoints) |
| `w600k_r50.onnx` | Recognition (`ArcFace ResNet50`) | 166.3 MB | Yes | **YES** | Retain (produces the 512-dim embedding) |
| `1k3d68.onnx` | 3D Landmark 68 | 137.0 MB | Yes | **NO** | **Exclude** (saves 137 MB disk / ~90 MB RAM) |
| `2d106det.onnx` | 2D Landmark 106 | 4.8 MB | Yes | **NO** | **Exclude** (saves 4.8 MB disk) |
| `genderage.onnx` | Gender & Age Estimation | 1.3 MB | Yes | **NO** | **Exclude** (saves 1.3 MB disk) |
| `buffalo_l.zip` | Archive | 281.0 MB | Cached on disk | **NO** | Delete after extraction |

### Measured Impact of Restricting Modules (`allowed_modules=['detection', 'recognition']`)

Empirical benchmark performed directly on the project environment:

| Metric | Unoptimized (Default `buffalo_l`) | Optimized (`detection` + `recognition` only) | Improvement |
| :--- | :--- | :--- | :--- |
| **Models Loaded in Memory** | 5 models | 2 models | -60% models |
| **Model Init Time (Cold)** | 3.72 s | 2.05 s | **45% faster** |
| **Inference Time (CPU)** | 6.92 s | 0.58 s | **11.9x faster (1200% speedup)** |
| **Working Set Memory (Init)** | 428 MB | 272 MB | **156 MB saved** |
| **Peak Working Set (Inference)**| 532 MB | 368 MB | **164 MB saved (safely fits 512MB RAM tier)** |
| **Embedding Compatibility** | Reference | Diff norm = 0.000000, Cosine sim = 1.000000 | **100% Bit-level match** |

---

## 5. Deployment Blockers Breakdown

### Blocker 1: Process Suicide via `exit()`
* **Location:** `backend/compare_faces.py:21, 26, 30`
* **Trigger:** When an image cannot be read, when zero faces are detected, or when more than 1 face is in the photo.
* **Impact:** The worker process terminates instantly. In a production server like Gunicorn, this triggers worker crash restarts (`SIGCHLD`), breaks concurrent requests, and sends a `502 Bad Gateway` to the user.
* **Fix:** Raise custom exceptions (e.g. `NoFaceDetectedError`, `MultipleFacesDetectedError`, `InvalidImageError`) and catch them in `app.py` to return clean JSON error payloads (`{"status": "no_face"}`).

### Blocker 2: Memory Exceeds 512 MB Free Tier
* **Location:** Default `FaceAnalysis()` initialization.
* **Impact:** Peak memory of 532 MB triggers the Linux cgroup OOM (Out Of Memory) killer on Render Free tier (512 MB cap) and small instances.
* **Fix:** Restrict `allowed_modules=['detection', 'recognition']`, reducing peak RAM to ~368 MB.

### Blocker 3: Disk I/O on Every HTTP Request
* **Location:** `backend/compare_faces.py:40-47`
* **Trigger:** Every `/upload` request iterates `os.listdir("embedding")` and calls `np.load` 8 times.
* **Impact:** Adds disk latency, file descriptor thrashing, and crashes if the working directory is not the project root.
* **Fix:** Load all celebrity embeddings into an in-memory dictionary once at startup.

### Blocker 4: Hardcoded `127.0.0.1:5000` in Frontend
* **Location:** `frontend/script.js:219, 331, 332`
* **Impact:** When deployed to the cloud, user browsers attempt to connect to their own local port 5000, causing network failure (`ERR_CONNECTION_REFUSED`).
* **Fix:** Use relative paths (e.g. `/upload`, `/dataset/`) when Flask serves frontend static assets, or support a configurable API base URL via `window.API_BASE_URL || ""`.

### Blocker 5: Case Sensitivity in File Names
* **Location:** `dataset/Anushka.jpg` vs `embedding/anushka.npy` and `app.py:50` (`f"/dataset/{best_match}.jpg"`).
* **Impact:** On Windows, filesystem lookups are case-insensitive. On Linux/Docker, requesting `/dataset/anushka.jpg` returns `404 Not Found`.
* **Fix:** Standardize all filenames in `dataset/` and `embedding/` to lowercase (e.g., `anushka.jpg`, `anushka.npy`), and build a dictionary mapping celebrity keys to actual filenames.

### Blocker 6: Unsanitized File Uploads & Unbounded Disk Growth
* **Location:** `backend/app.py:41-43`
* **Impact:** Uses raw `image.filename` directly (path traversal risk). Files accumulate indefinitely in `uploads/` (16.4 MB of test images already stored), eventually exhausting ephemeral container storage.
* **Fix:** Use `werkzeug.utils.secure_filename`, generate unique UUID-based temporary filenames, and delete temporary uploaded files after embedding extraction.

### Blocker 7: Non-Binding Development Server
* **Location:** `backend/app.py:57` (`if __name__ == "__main__": app.run(debug=True)`)
* **Impact:** Flask dev server binds only to `127.0.0.1` and ignores the cloud platform's `$PORT` environment variable.
* **Fix:** Add dynamic port binding (`int(os.environ.get("PORT", 5000))`), `host="0.0.0.0"`, and provide a production WSGI entry point for Gunicorn (`gunicorn backend.app:app`).

---

## 6. Recommended Action Plan

1. **Backend Refactoring (`backend/compare_faces.py` & `backend/app.py`):**
   - Replace `exit()` calls with structured domain exceptions.
   - Restrict `FaceAnalysis` to `allowed_modules=['detection', 'recognition']`.
   - Explicitly configure CPU provider (`providers=['CPUExecutionProvider']`, `ctx_id=-1`).
   - Cache celebrity embeddings in memory at application startup.
   - Anchor all filesystem paths to `BASE_DIR`.
   - Sanitize uploads with `secure_filename` and auto-cleanup temporary files.
   - Standardize celebrity image resolution to avoid case-sensitivity bugs.
   - Bind host/port to environment variables (`PORT`, `HOST`).

2. **Frontend Adjustments (`frontend/script.js` & `frontend/index.html`):**
   - Rename `index (1).html` back to clean `index.html`.
   - Remove hardcoded `http://127.0.0.1:5000` in favor of dynamic/relative paths.
   - Enable Flask to optionally serve the frontend static files directly (unified deployment mode) or keep it cleanly decoupled for static hosting (decoupled mode).

3. **Dependency Pruning & Configuration:**
   - Prune `requirements.txt`: remove `pandas`, `pillow`, switch to `opencv-python-headless`.
   - Re-encode `requirements.txt` to UTF-8 without BOM.
   - Add `.dockerignore` and update `.gitignore` to prevent tracking `uploads/`, `venv/`, and cache artifacts.
   - Create production-ready `Dockerfile` optimized for CPU inference with minimal layer size.

4. **Testing & Verification:**
   - Verify zero regression against existing `.npy` embeddings.
   - Verify error responses (`no_face`, `multiple_faces`, `invalid_image`).
   - Measure actual RAM, startup time, and inference speed.
