# Use official lightweight Python slim image
FROM python:3.11-slim

# Set environment flags
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000 \
    HOST=0.0.0.0 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    INSIGHTFACE_ROOT=/root/.insightface

# Install minimal OS dependencies for ONNX Runtime (libgomp1 for OpenMP thread pool)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download InsightFace buffalo_l models during build to avoid cold-start runtime latency
# Restrict allowed modules and delete unneeded 3D/landmark models and the zip to keep image small
RUN python -c "\
from insightface.app import FaceAnalysis; \
app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'recognition'], providers=['CPUExecutionProvider']); \
app.prepare(ctx_id=-1)" && \
    rm -f /root/.insightface/models/buffalo_l.zip \
          /root/.insightface/models/buffalo_l/1k3d68.onnx \
          /root/.insightface/models/buffalo_l/2d106det.onnx \
          /root/.insightface/models/buffalo_l/genderage.onnx

# Copy application source code
COPY . .

# Ensure upload directory exists
RUN mkdir -p /app/uploads

# Expose port (default 5000, overridden dynamically by cloud platforms)
EXPOSE 5000

# Healthcheck probe
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Start production server with 1 worker and 4 threads to strictly constrain memory under 512 MB
CMD exec gunicorn --bind "0.0.0.0:${PORT}" --workers 1 --threads 4 --timeout 120 backend.app:app
