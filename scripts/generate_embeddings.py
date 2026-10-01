#!/usr/bin/env python3
"""
scripts/generate_embeddings.py
Incremental face embedding generator for Hamshakal Finder.

Supports:
  python scripts/generate_embeddings.py --new
  python scripts/generate_embeddings.py --celebrity <key>
  python scripts/generate_embeddings.py --all [--force]
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATASET_DIR = PROJECT_ROOT / "dataset"
EMBEDDING_DIR = PROJECT_ROOT / "embedding"
METADATA_FILE = PROJECT_ROOT / "dataset" / "metadata" / "celebrities.json"
REPORTS_DIR = PROJECT_ROOT / "validation_reports"

os.makedirs(EMBEDDING_DIR, exist_ok=True)
os.makedirs(METADATA_FILE.parent, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def get_face_analyzer():
    """Initializes and returns CPU-optimized InsightFace FaceAnalysis instance."""
    from insightface.app import FaceAnalysis

    models_root = os.environ.get("INSIGHTFACE_ROOT", os.path.expanduser("~/.insightface"))
    model_name = os.environ.get("INSIGHTFACE_MODEL_NAME", "buffalo_s")

    print(f"[Embedding Gen] Initializing FaceAnalysis ({model_name}) on CPU...")
    app = FaceAnalysis(
        name=model_name,
        root=models_root,
        allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"]
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def load_metadata() -> Dict:
    """Loads dataset/metadata/celebrities.json or returns default template."""
    if METADATA_FILE.exists():
        try:
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Warning] Failed to parse {METADATA_FILE}: {e}")
    return {
        "version": "1.0.0",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "celebrities": {}
    }


def save_metadata(metadata: Dict):
    """Saves updated metadata to dataset/metadata/celebrities.json atomically."""
    metadata["updated_at"] = datetime.now(timezone.utc).isoformat()
    temp_file = METADATA_FILE.with_suffix(".tmp")
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    temp_file.replace(METADATA_FILE)


def discover_celebrity_images() -> Dict[str, Path]:
    """
    Discovers all candidate celebrity images across:
    1. metadata entries in celebrities.json
    2. flat images in dataset/*.jpg
    3. nested folders in dataset/celebrities/<key>/*.*
    Returns dictionary mapping lowercased celebrity_key -> image Path.
    """
    candidates = {}

    # 1. From metadata
    metadata = load_metadata()
    for key, data in metadata.get("celebrities", {}).items():
        img_p = data.get("image_path")
        if img_p:
            p = PROJECT_ROOT / img_p
            if p.exists() and p.is_file():
                candidates[key.lower()] = p

    # 2. From flat dataset directory
    if DATASET_DIR.exists():
        for file in DATASET_DIR.iterdir():
            if file.is_file() and file.suffix.lower() in ALLOWED_EXTENSIONS:
                key = file.stem.lower()
                if key not in candidates:
                    candidates[key] = file

    # 3. From dataset/celebrities/ subdirectories
    celeb_dir = DATASET_DIR / "celebrities"
    if celeb_dir.exists() and celeb_dir.is_dir():
        for folder in celeb_dir.iterdir():
            if folder.is_dir():
                key = folder.name.lower()
                for img_file in folder.iterdir():
                    if img_file.is_file() and img_file.suffix.lower() in ALLOWED_EXTENSIONS:
                        candidates[key] = img_file
                        break

    return candidates


def validate_existing_embedding(emb_path: Path) -> bool:
    """Verifies that an existing .npy file is a valid 512-dim float32 array without NaNs."""
    if not emb_path.exists():
        return False
    try:
        arr = np.load(str(emb_path))
        if arr.ndim == 1 and arr.shape[0] == 512 and not np.isnan(arr).any():
            return True
    except Exception:
        pass
    return False


def process_image(
    analyzer,
    image_path: Path
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Decodes an image and extracts a normalized 512-dim ArcFace embedding.
    Returns (embedding, None) on success or (None, failure_reason) on failure.
    """
    image = cv2.imread(str(image_path))
    if image is None or image.size == 0:
        return None, "Invalid image format or unreadable file"

    faces = analyzer.get(image)
    if len(faces) == 0:
        return None, "No face detected in reference photo"

    if len(faces) > 1:
        return None, f"Multiple faces detected ({len(faces)}). Reference photo must contain exactly one face"

    emb = faces[0].embedding
    if emb is None or emb.shape != (512,):
        return None, f"Unexpected embedding shape: {getattr(emb, 'shape', None)}"

    # Normalize embedding to unit L2 norm
    norm = np.linalg.norm(emb)
    if norm == 0:
        return None, "Embedding vector has zero norm"
    emb = emb / norm

    return emb.astype(np.float32), None


def run_embedding_generation(
    mode: str = "new",
    target_celebrity: Optional[str] = None,
    force: bool = False
) -> Dict:
    """
    Main generator execution.
    Modes:
      - 'new': Only images without valid .npy embeddings
      - 'celebrity': Single celebrity matching target_celebrity
      - 'all': All discovered images
    """
    candidates = discover_celebrity_images()
    metadata = load_metadata()
    celeb_meta = metadata.setdefault("celebrities", {})

    total_records = len(candidates)
    if target_celebrity:
        key = target_celebrity.lower()
        if key not in candidates:
            print(f"[Error] Celebrity '{target_celebrity}' not found in dataset images or metadata.")
            return {"status": "error", "message": f"Celebrity {target_celebrity} not found"}
        candidates = {key: candidates[key]}

    # Determine which celebrities need processing
    to_process = {}
    already_processed_count = 0

    for key, img_path in candidates.items():
        emb_file = EMBEDDING_DIR / f"{key}.npy"
        if not force and validate_existing_embedding(emb_file):
            already_processed_count += 1
            if mode == "new":
                continue
        to_process[key] = (img_path, emb_file)

    print("\n" + "=" * 60)
    print(f" HAMSHAKAL FINDER — EMBEDDING GENERATOR (Mode: {mode.upper()})")
    print("=" * 60)
    print(f"Total discovered candidates : {total_records}")
    print(f"Already validly processed   : {already_processed_count}")
    print(f"Items queued for extraction : {len(to_process)}")
    print("=" * 60)

    if not to_process:
        print("[Notice] All embeddings are up to date! Nothing to generate.")
        return {
            "total_candidates": total_records,
            "already_processed": already_processed_count,
            "queued": 0,
            "successfully_generated": 0,
            "success": 0,
            "failed": 0,
            "failures": {}
        }

    # Initialize model once
    t0 = time.time()
    analyzer = get_face_analyzer()

    success_count = 0
    failures = {}

    for key, (img_path, emb_file) in to_process.items():
        emb, err = process_image(analyzer, img_path)
        if err is not None:
            failures[key] = err
            print(f"  [FAIL] [{key}] FAILED: {err}")
            continue

        # Save embedding atomically
        temp_emb = emb_file.with_name(f"{emb_file.stem}_temp.npy")
        np.save(str(temp_emb), emb)
        temp_emb.replace(emb_file)

        # Update metadata entry
        rel_img = str(img_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        rel_emb = str(emb_file.relative_to(PROJECT_ROOT)).replace("\\", "/")

        record = celeb_meta.setdefault(key, {})
        record["celebrity_key"] = key
        if "celebrity_name" not in record:
            record["celebrity_name"] = key.replace("_", " ").title()
        record["image_path"] = rel_img
        record["image_filename"] = img_path.name
        record["embedding_path"] = rel_emb
        record["status"] = "active"
        record["last_embedded_at"] = datetime.now(timezone.utc).isoformat()

        success_count += 1
        print(f"  [SUCCESS] [{key}] Generated embedding: {emb_file.name}")

    save_metadata(metadata)
    elapsed = time.time() - t0

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "total_records": total_records,
        "already_processed": already_processed_count,
        "queued": len(to_process),
        "successfully_generated": success_count,
        "failed": len(failures),
        "failures": failures,
        "elapsed_seconds": round(elapsed, 2)
    }

    report_path = REPORTS_DIR / "embedding_generation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "-" * 60)
    print(" GENERATION SUMMARY")
    print("-" * 60)
    print(f"Successfully generated : {success_count}")
    print(f"Failed                 : {len(failures)}")
    print(f"Elapsed time           : {elapsed:.2f}s")
    if failures:
        print("\nFailure Details:")
        for k, reason in failures.items():
            print(f"  - {k}: {reason}")
    print(f"\nDetailed report saved to: {report_path.relative_to(PROJECT_ROOT)}")
    print("-" * 60)

    return report


def main():
    parser = argparse.ArgumentParser(description="Incremental Face Embedding Generator for Hamshakal Finder")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--new", action="store_true", help="Process only new images that do not yet have valid embeddings")
    group.add_argument("--celebrity", type=str, metavar="KEY", help="Process or reprocess a single celebrity by key")
    group.add_argument("--all", action="store_true", help="Process all celebrity images")
    parser.add_argument("--force", action="store_true", help="Force regeneration even if embedding already exists")

    args = parser.parse_args()

    if args.new:
        run_embedding_generation(mode="new", force=args.force)
    elif args.celebrity:
        run_embedding_generation(mode="celebrity", target_celebrity=args.celebrity, force=args.force)
    elif args.all:
        run_embedding_generation(mode="all", force=args.force)


if __name__ == "__main__":
    main()
