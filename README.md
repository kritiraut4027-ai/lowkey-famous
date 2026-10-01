# 🎭 Hamshakal Finder

An AI-powered celebrity lookalike finder that compares facial geometry against an indexed celebrity dataset using deep ArcFace facial embeddings and cosine similarity.

---

## 🚀 Features

* **High-Accuracy Face Recognition:** Powered by InsightFace SCRFD face detector and ArcFace ResNet-50 embedding model (512-dimensional vector space).
* **Lightweight & Resource-Optimized:** Specifically engineered to run in low-resource environments (reduced from 532 MB peak RAM down to **368 MB**, fitting within standard 512 MB free-tier hosting).
* **10x Faster CPU Inference:** Restricts model loading strictly to detection and recognition, cutting inference time from 6.9 seconds down to **0.58 seconds** on standard CPU.
* **In-Memory Caching:** Celebrity embeddings are loaded once at startup into memory, eliminating per-request disk I/O.
* **Privacy & Storage Hygiene:** Uploaded user images are processed in-memory / temporary files and deleted automatically after embedding extraction.
* **Dual Deployment Architecture:** Can be deployed as a unified full-stack application (Flask serving frontend + API) or cleanly decoupled (Vercel static frontend + Render/Hugging Face backend).
* **Robust Error Handling:** Replaced crashing `exit()` calls with structured JSON domain errors (`no_face`, `multiple_faces`, `invalid_image`).

---

## 📁 Project Structure

```text
Hamshakal-finder/
├── backend/
│   ├── app.py                  # Production Flask application & API routes
│   └── compare_faces.py        # Core ML pipeline, ArcFace singleton, similarity math
├── dataset/                    # Reference celebrity images (.jpg)
├── embedding/                  # Pre-computed 512-d float32 embeddings (.npy)
├── frontend/                   # Responsive single-page web application
│   ├── index.html              # Modern dark-mode UI
│   ├── script.js               # Dynamic upload, scan animation, API caller
│   └── style.css               # Vanilla CSS design system
├── scripts/
│   └── generating_embedding.py # Offline script to regenerate .npy embeddings
├── test_suite.py               # Comprehensive 14-point test suite
├── Dockerfile                  # Production container definition (CPU-optimized)
├── requirements.txt            # Minimal pinned dependencies (clean UTF-8)
├── DEPLOYMENT_AUDIT.md         # Full audit report of initial bottlenecks
├── DEPLOYMENT_TARGETS.md       # Comparative platform analysis (Render, HF, Vercel)
└── RESOURCE_USAGE.md           # Empirical memory, CPU, and storage benchmarks
```

---

## ⚙️ Local Development Setup

### 1. Prerequisites
* Python 3.10 – 3.12 (or 3.13)
* Git

### 2. Clone and Setup Environment
```bash
git clone https://github.com/your-username/Hamshakal-finder.git
cd Hamshakal-finder

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Test Suite
```bash
python test_suite.py
```
*All 14 unit and integration tests should pass.*

### 4. Start the Local Server
```bash
python backend/app.py
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## 🐳 Docker Deployment

The included `Dockerfile` pre-downloads the required models (`det_10g.onnx` and `w600k_r50.onnx`) during build time and strips unneeded 3D/landmark weights to ensure instant container startup and small footprint.

### Build and Run Locally:
```bash
docker build -t hamshakal-finder .
docker run -p 5000:5000 -e PORT=5000 hamshakal-finder
```
Visit [http://localhost:5000](http://localhost:5000).

---

## ☁️ Free Cloud Deployment Guides

### Option 1: Hugging Face Spaces (Top Recommended Free Hosting)
Hugging Face Spaces offers a permanent **Free CPU Basic Tier with 16 GB RAM and 2 vCPUs**, with no idle sleep.

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. Select **Docker** as the Space SDK and **Blank** template.
3. Push this repository to your Hugging Face Space repository:
   ```bash
   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/hamshakal-finder
   git push space main
   ```
4. Hugging Face will automatically build the `Dockerfile` and launch your live application at `https://your-username-hamshakal-finder.hf.space`.

---

### Option 2: Render Free Web Service
Render provides a free web service tier with 512 MB RAM.

1. Create a new **Web Service** on [Render](https://render.com).
2. Connect your GitHub repository.
3. Select **Docker** as the runtime.
4. Set the start command / environment variables:
   * `PORT`: `5000`
5. Deploy.
> **Note:** Render free instances spin down after 15 minutes of inactivity. When cold-starting, the first request takes ~45 seconds. Thanks to our optimizations, peak RAM is 368 MB, safely below Render's 512 MB kill threshold.

---

### Option 3: Decoupled (Vercel Frontend + HF / Render Backend)
If you prefer your frontend on Vercel's global CDN:
1. Deploy the `frontend/` directory to Vercel.
2. In `frontend/script.js` or via a small `<script>` tag before `script.js` in `index.html`:
   ```html
   <script>window.API_BASE_URL = "https://your-backend.hf.space";</script>
   ```
3. The Flask backend already has full CORS enabled to handle requests from any origin.

---

## 📡 API Reference

### 1. Healthcheck
* **Endpoint:** `GET /health`
* **Response:**
  ```json
  {
    "status": "healthy",
    "celebrities_indexed": 8
  }
  ```

### 2. Face Match
* **Endpoint:** `POST /upload`
* **Content-Type:** `multipart/form-data`
* **Body:** `image` (binary file: JPG, PNG, WEBP, max 8 MB)
* **Success Response (HTTP 200):**
  ```json
  {
    "status": "ok",
    "celebrity_key": "alia",
    "celebrity_name": "Alia Bhatt",
    "celebrity_image_url": "/dataset/alia.jpg",
    "match_percentage": 94.25,
    "similarity_score": 0.9425,
    "score_type": "cosine_similarity"
  }
  ```
* **Error Responses:**
  * `{"status": "no_face", "error": "No face detected in the image."}` (HTTP 200)
  * `{"status": "multiple_faces", "error": "Multiple faces detected..."}` (HTTP 200)
  * `{"status": "invalid_image", "error": "Unsupported file format..."}` (HTTP 400)

---

## 🔒 Privacy & Biometric Data Handling

* Uploaded user photos are used solely to extract the facial embedding vector for the duration of the HTTP request.
* Temporary image files are deleted immediately after vector extraction via a guaranteed `finally` block in `app.py`.
* No user facial embeddings or photos are persisted to databases or long-term disk storage.
