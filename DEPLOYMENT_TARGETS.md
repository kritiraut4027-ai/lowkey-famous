# Deployment Target Analysis: Hamshakal Finder MVP

**Date:** September 2026  
**Architect:** Senior ML Engineer & Backend Deployment Architect  
**Objective:** Evaluate free and low-cost cloud platforms for hosting Hamshakal Finder's CPU-based face recognition workload.

---

## 1. Candidate Platform Comparison Matrix

| Platform | Free Tier Available? | RAM Limits | CPU Limits | Ephemeral Storage | Idle Sleep Behavior | Build Limits | Docker Supported? | Can Host Model? | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Hugging Face Spaces** | **YES** (Free CPU Basic) | **16 GB RAM** | 2 vCPU | 50 GB | **No sleeping** (always warm) | 30 min build timeout | Yes (Docker or Gradio SDK) | **YES** (Effortless) | ⭐ **#1 Top Recommendation (Best Free Tier)** |
| **Render** | **YES** (Free Web Service) | **512 MB RAM** | 0.5 vCPU shared | Ephemeral, ~1-2 GB | **Sleeps after 15 min** of inactivity; ~50s cold start | 15 min build timeout | Yes (Native or Docker) | **YES** (With our optimizations: 368MB < 512MB) | **Feasible Free Tier (Requires 1 worker)** |
| **Railway** | **NO FREE TIER** ($5 trial credit only) | 512 MB – 8 GB | 1 – 8 vCPU | Flexible | Configurable sleep | Generous | Yes | **YES** | **Paid ($5/mo minimum usage)** |
| **Vercel** | **YES** (Hobby Tier) | **1024 MB RAM** | Shared vCPU | 500 MB `/tmp` | **Serverless invocation** | 45-second function timeout; 50 MB / 250 MB unzipped bundle limit | No (Serverless functions only) | **NO** (Bundle size & cold start exceed limits) | ❌ **UNSUITABLE for Backend Inference** |
| **Koyeb** | **YES** (Eco Free Tier) | **512 MB RAM** | 0.1 vCPU | 2 GB | Does not sleep on free tier | Standard | Yes (Docker) | **YES** (With our optimizations) | **Good Alternative** |

---

## 2. In-Depth Platform Evaluations

### 2.1 Hugging Face Spaces (Top Recommendation: Best Free Tier for ML)
* **Free Tier Availability:** Permanent free CPU Basic tier with **16 GB RAM and 2 vCPUs**.
* **Storage Limitations:** 50 GB free disk space.
* **RAM Limitations:** 16 GB RAM (far exceeding the 368 MB required by our optimized model).
* **CPU Limitations:** 2 vCPU cores, providing fast inference (~0.35s – 0.55s).
* **Sleep Behavior:** Free Spaces do not sleep by default for active spaces (or can be configured to remain active without cold start).
* **Build Limitations:** Supports custom `Dockerfile`.
* **Workload Permission:** Designed specifically for hosting ML models and demos.
* **Architecture Suitability:** **PERFECT**. The app runs with zero risk of OOM, supports both frontend and backend in one space, and costs $0.

---

### 2.2 Render (Free Web Service)
* **Free Tier Availability:** 1 free web service per account (750 free hours/month).
* **RAM Limitations:** **Strict 512 MB RAM limit**.
  - *Prior to optimization:* The app consumed 532 MB peak RAM, causing Render to instantly kill the container via OOM (`Exit code 137`).
  - *With our optimization:* Peak memory is reduced to **368 MB**, allowing the app to successfully run inside Render's 512 MB limit.
* **CPU Limitations:** Shared 0.5 vCPU. Inference takes ~0.7s to 1.2s.
* **Storage Limitations:** Ephemeral filesystem (resets on deploy). Our Docker build pre-caches the 182 MB model weights, avoiding cold-start downloads.
* **Sleep Behavior:** Spins down after 15 minutes of inactivity. When a user visits after idle, Render takes 45–60 seconds to spin up the container.
* **Critical Requirement:** Must use `gunicorn --workers 1 --threads 4`. Spawning 2 or more workers will exceed 512 MB RAM and crash the service.

---

### 2.3 Railway
* **Free Tier Status:** Railway no longer offers a permanent free tier. New users receive a one-time $5 trial credit. Ongoing hosting requires a $5/month minimum plan.
* **Capabilities:** Unlimited custom Docker containers, customizable RAM and CPU allocation.
* **Verdict:** Highly reliable, but **not free long-term**. If zero cost is a strict requirement, avoid Railway in favor of Hugging Face Spaces or Render.

---

### 2.4 Vercel (Why Serverless Fails for InsightFace)
* **Bundle Limit:** Vercel serverless functions have a maximum uncompressed deployment size of 250 MB (and 50 MB compressed). Our minimal runtime dependencies (`site-packages` ~395 MB) + model weights (`w600k_r50.onnx` + `det_10g.onnx` = 182 MB) total **~577 MB**, exceeding Vercel's limit by over 2x.
* **Function Cold Start:** Loading ONNX models on cold serverless spin-up takes 2–4 seconds per cold invocation.
* **Feasible Vercel Architecture:** Deploy **only the frontend static files** (`index.html`, `style.css`, `script.js`) to Vercel (free, instant global CDN), and point `window.API_BASE_URL` to a backend hosted on Hugging Face Spaces or Render.

---

## 3. Recommended Architectural Deployments

### Option 1: Unified Full-Stack on Hugging Face Spaces (Simplest & Best Free Tier)
* Deploy the entire repository (Docker-based) directly to Hugging Face Spaces.
* Flask serves both the static frontend UI (`/`) and the API (`/upload`, `/health`, `/dataset/<filename>`).
* 16 GB RAM ensures rock-solid stability and zero OOM risk.

### Option 2: Decoupled (Vercel Frontend + Render / HF Backend)
* **Frontend:** Deployed to Vercel / Netlify / GitHub Pages for free static hosting with zero cold start.
* **Backend:** Deployed to Hugging Face Spaces or Render Free Tier as an API.
* In `script.js`, set `window.API_BASE_URL = "https://your-backend.hf.space";`. CORS is already configured and enabled on all routes.
