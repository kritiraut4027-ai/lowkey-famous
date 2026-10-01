#!/usr/bin/env python3
"""
scripts/add_celebrity.py
Command-line utility to register, validate, and embed a new celebrity in Hamshakal Finder.

Usage:
  python scripts/add_celebrity.py --key <key> --name <display_name> --image <image_path> [--gender male|female] [--tags "tag1,tag2"]
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from backend.compare_faces import get_face_analyzer

DATASET_DIR = PROJECT_ROOT / "dataset"
EMBEDDING_DIR = PROJECT_ROOT / "embedding"
METADATA_FILE = PROJECT_ROOT / "dataset" / "metadata" / "celebrities.json"

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(EMBEDDING_DIR, exist_ok=True)
os.makedirs(METADATA_FILE.parent, exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def add_celebrity(
    key: str,
    name: str,
    image_src: str,
    gender: Optional[str] = None,
    tags: Optional[str] = None,
    use_nested_folder: bool = False
) -> bool:
    key = key.strip().lower().replace(" ", "_")
    src_path = Path(image_src).resolve()

    print("\n" + "=" * 60)
    print(f" HAMSHAKAL FINDER -- REGISTER NEW CELEBRITY: {key}")
    print("=" * 60)

    # 1. Validate Source Image File
    if not src_path.exists():
        print(f"[Error] Source image file does not exist: {src_path}")
        return False

    if src_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        print(f"[Error] Unsupported image extension '{src_path.suffix}'. Allowed: {ALLOWED_EXTENSIONS}")
        return False

    img = cv2.imread(str(src_path))
    if img is None or img.size == 0:
        print("[Error] Image cannot be read or is corrupted.")
        return False

    # 2. Run Face Detection Check
    print("[*] Inspecting facial geometry and face count...")
    analyzer = get_face_analyzer()
    faces = analyzer.get(img)

    if len(faces) == 0:
        print("[Error] No face detected in the provided image. Please use a clearer portrait.")
        return False

    if len(faces) > 1:
        print(f"[Error] Multiple faces ({len(faces)}) detected. Reference photos must contain exactly one face.")
        return False

    user_emb = faces[0].embedding
    if user_emb is None or user_emb.shape != (512,):
        print(f"[Error] Failed to extract 512-dim embedding. Got shape: {getattr(user_emb, 'shape', None)}")
        return False

    # Normalize embedding
    norm = np.linalg.norm(user_emb)
    if norm > 0:
        user_emb = user_emb / norm
    user_emb = user_emb.astype(np.float32)

    # 3. Determine Destination Path
    ext = src_path.suffix.lower()
    if use_nested_folder:
        dest_folder = DATASET_DIR / "celebrities" / key
        dest_folder.mkdir(parents=True, exist_ok=True)
        dest_img_path = dest_folder / f"image{ext}"
    else:
        dest_img_path = DATASET_DIR / f"{key}{ext}"

    # Copy image
    shutil.copy2(str(src_path), str(dest_img_path))
    print(f"[+] Image saved to: {dest_img_path.relative_to(PROJECT_ROOT)}")

    # 4. Save Embedding (.npy)
    emb_dest_path = EMBEDDING_DIR / f"{key}.npy"
    np.save(str(emb_dest_path), user_emb)
    print(f"[+] Embedding saved to: {emb_dest_path.relative_to(PROJECT_ROOT)} (512 float32)")

    # 5. Update Metadata Registry (celebrities.json)
    metadata = {"version": "1.0.0", "celebrities": {}}
    if METADATA_FILE.exists():
        try:
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            pass

    rel_img_path = str(dest_img_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    rel_emb_path = str(emb_dest_path.relative_to(PROJECT_ROOT)).replace("\\", "/")

    parsed_tags = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

    record = {
        "celebrity_key": key,
        "celebrity_name": name.strip(),
        "image_path": rel_img_path,
        "image_filename": dest_img_path.name,
        "embedding_path": rel_emb_path,
        "status": "active",
        "gender": gender.strip().lower() if gender else "unspecified",
        "tags": parsed_tags,
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    metadata.setdefault("celebrities", {})[key] = record
    metadata["updated_at"] = datetime.now(timezone.utc).isoformat()

    temp_meta = METADATA_FILE.with_suffix(".tmp")
    with open(temp_meta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    temp_meta.replace(METADATA_FILE)

    print(f"[+] Metadata updated in: {METADATA_FILE.relative_to(PROJECT_ROOT)}")
    print("=" * 60)
    print(f"[SUCCESS] '{name}' (key: {key}) is now fully indexed and active!")
    print("=" * 60 + "\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="Add and index a new celebrity in Hamshakal Finder")
    parser.add_argument("--key", required=True, help="Unique alphanumeric key, e.g. ryan_gosling")
    parser.add_argument("--name", required=True, help="Formatted display name, e.g. 'Ryan Gosling'")
    parser.add_argument("--image", required=True, help="Path to input photo (JPG, PNG, WEBP)")
    parser.add_argument("--gender", default=None, choices=["male", "female", "other"], help="Gender category")
    parser.add_argument("--tags", default=None, help="Comma-separated tags, e.g. 'hollywood,actor'")
    parser.add_argument("--nested", action="store_true", help="Store in dataset/celebrities/<key>/ instead of flat dataset/")

    args = parser.parse_args()

    success = add_celebrity(
        key=args.key,
        name=args.name,
        image_src=args.image,
        gender=args.gender,
        tags=args.tags,
        use_nested_folder=args.nested
    )
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
