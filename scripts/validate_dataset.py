#!/usr/bin/env python3
"""
scripts/validate_dataset.py
Comprehensive Dataset Quality & Integrity Auditor for Hamshakal Finder.

Checks:
- Supported image formats and readability
- Corrupted images and empty files
- Face detection count (flags images with 0 or >1 face)
- Embedding dimension compatibility (512-dim float32)
- Embedding validity (no NaNs, non-zero norm)
- Metadata completeness and duplicate keys
- Orphaned images and embeddings
- Outputs JSON audit report to validation_reports/validation_report.json
"""

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set, Any

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

os.makedirs(REPORTS_DIR, exist_ok=True)
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def calculate_image_hash(image_path: Path) -> str:
    """Computes MD5 hash of an image file to detect duplicate images."""
    try:
        hasher = hashlib.md5()
        with open(image_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return ""


def run_dataset_validation(verify_faces: bool = True) -> Dict[str, Any]:
    print("\n" + "=" * 65)
    print(" HAMSHAKAL FINDER — DATASET QUALITY & INTEGRITY AUDITOR")
    print("=" * 65)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_celebrities_registered": 0,
            "valid_records": 0,
            "records_with_issues": 0,
            "total_images_found": 0,
            "total_embeddings_found": 0,
            "orphaned_images": [],
            "orphaned_embeddings": [],
            "duplicate_image_hashes": {}
        },
        "details": {}
    }

    # 1. Load Metadata
    metadata = {}
    if METADATA_FILE.exists():
        try:
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                metadata = data.get("celebrities", {})
                print(f"[OK] Metadata file loaded ({len(metadata)} entries registered)")
        except Exception as e:
            print(f"[FAIL] Could not parse metadata file {METADATA_FILE}: {e}")
    else:
        print(f"[Notice] Metadata file {METADATA_FILE.relative_to(PROJECT_ROOT)} not found")

    report["summary"]["total_celebrities_registered"] = len(metadata)

    # 2. Discover physical images and embeddings on disk
    physical_images: Dict[str, Path] = {}
    image_hashes: Dict[str, List[str]] = {}

    if DATASET_DIR.exists():
        for f in DATASET_DIR.iterdir():
            if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS:
                k = f.stem.lower()
                physical_images[k] = f
                h = calculate_image_hash(f)
                if h:
                    image_hashes.setdefault(h, []).append(str(f.relative_to(PROJECT_ROOT)))

        celeb_dir = DATASET_DIR / "celebrities"
        if celeb_dir.exists() and celeb_dir.is_dir():
            for folder in celeb_dir.iterdir():
                if folder.is_dir():
                    k = folder.name.lower()
                    for img in folder.iterdir():
                        if img.is_file() and img.suffix.lower() in ALLOWED_EXTENSIONS:
                            physical_images[k] = img
                            h = calculate_image_hash(img)
                            if h:
                                image_hashes.setdefault(h, []).append(str(img.relative_to(PROJECT_ROOT)))
                            break

    physical_embeddings: Dict[str, Path] = {}
    if EMBEDDING_DIR.exists():
        for f in EMBEDDING_DIR.iterdir():
            if f.is_file() and f.suffix.lower() == ".npy":
                k = f.stem.lower()
                physical_embeddings[k] = f

    report["summary"]["total_images_found"] = len(physical_images)
    report["summary"]["total_embeddings_found"] = len(physical_embeddings)

    # Detect duplicate images
    for h, paths in image_hashes.items():
        if len(paths) > 1:
            report["summary"]["duplicate_image_hashes"][h] = paths

    # Aggregate all unique keys to inspect
    all_keys = set(metadata.keys()).union(physical_images.keys()).union(physical_embeddings.keys())

    # Optional: Load FaceAnalysis for face detection check
    analyzer = None
    if verify_faces:
        try:
            from backend.compare_faces import get_face_analyzer
            analyzer = get_face_analyzer()
            print("[OK] FaceAnalysis detector initialized for face count validation")
        except Exception as e:
            print(f"[Warning] Could not initialize FaceAnalysis for validation: {e}")

    print("\nAuditing individual records:")
    print("-" * 65)

    valid_count = 0
    issue_count = 0

    for key in sorted(all_keys):
        issues = []
        record_info = metadata.get(key, {})

        # Check metadata
        if key not in metadata:
            issues.append("Missing from celebrities.json metadata registry")

        # Check image existence
        img_path = physical_images.get(key)
        if not img_path:
            issues.append("Reference image not found on disk")
        else:
            # Check image readability and dimensions
            img = cv2.imread(str(img_path))
            if img is None:
                issues.append("Image file is corrupted or unreadable by OpenCV")
            else:
                h, w = img.shape[:2]
                if h < 80 or w < 80:
                    issues.append(f"Image resolution too small: {w}x{h} (minimum 80x80 recommended)")

                # Validate face count if analyzer is active
                if analyzer is not None:
                    try:
                        faces = analyzer.get(img)
                        if len(faces) == 0:
                            issues.append("No face detected in reference photo")
                        elif len(faces) > 1:
                            issues.append(f"Multiple faces ({len(faces)}) detected in reference photo")
                    except Exception as e:
                        issues.append(f"Face detection error: {e}")

        # Check embedding existence and integrity
        emb_path = physical_embeddings.get(key)
        if not emb_path:
            issues.append(f"Missing embedding file: embedding/{key}.npy")
        else:
            try:
                emb = np.load(str(emb_path))
                if emb.ndim != 1 or emb.shape[0] != 512:
                    issues.append(f"Invalid embedding dimension: {emb.shape} (expected (512,))")
                if emb.dtype != np.float32 and emb.dtype != np.float64:
                    issues.append(f"Unexpected embedding dtype: {emb.dtype} (expected float32)")
                if np.isnan(emb).any():
                    issues.append("Embedding vector contains NaN values")
                if np.isinf(emb).any():
                    issues.append("Embedding vector contains Inf values")
                norm = np.linalg.norm(emb)
                if norm == 0:
                    issues.append("Embedding vector has zero norm")
            except Exception as e:
                issues.append(f"Could not load .npy embedding file: {e}")

        # Record results
        status_label = "VALID" if not issues else "ISSUES DETECTED"
        if not issues:
            valid_count += 1
            icon = "[PASS]"
        else:
            issue_count += 1
            icon = "[WARN]"

        print(f"  {icon} {key:<16} : {status_label}")
        if issues:
            for issue in issues:
                print(f"     - {issue}")

        report["details"][key] = {
            "status": "valid" if not issues else "invalid",
            "issues": issues,
            "has_metadata": key in metadata,
            "has_image": img_path is not None,
            "image_path": str(img_path.relative_to(PROJECT_ROOT)).replace("\\", "/") if img_path else None,
            "has_embedding": emb_path is not None,
            "embedding_path": str(emb_path.relative_to(PROJECT_ROOT)).replace("\\", "/") if emb_path else None
        }

    # Find orphaned files
    report["summary"]["orphaned_images"] = [
        str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")
        for k, p in physical_images.items()
        if k not in physical_embeddings
    ]
    report["summary"]["orphaned_embeddings"] = [
        str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")
        for k, p in physical_embeddings.items()
        if k not in physical_images
    ]

    report["summary"]["valid_records"] = valid_count
    report["summary"]["records_with_issues"] = issue_count

    # Save report
    out_file = REPORTS_DIR / "validation_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 65)
    print(" VALIDATION AUDIT SUMMARY")
    print("=" * 65)
    print(f"Total entries audited   : {len(all_keys)}")
    print(f"Fully valid entries     : {valid_count}")
    print(f"Entries with issues     : {issue_count}")
    print(f"Orphaned images         : {len(report['summary']['orphaned_images'])}")
    print(f"Orphaned embeddings     : {len(report['summary']['orphaned_embeddings'])}")
    print(f"Duplicate image hashes  : {len(report['summary']['duplicate_image_hashes'])}")
    print(f"\nFull JSON report saved to: {out_file.relative_to(PROJECT_ROOT)}")
    print("=" * 65 + "\n")

    return report


if __name__ == "__main__":
    run_dataset_validation(verify_faces=True)
