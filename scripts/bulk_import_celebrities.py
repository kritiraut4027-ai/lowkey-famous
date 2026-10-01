#!/usr/bin/env python3
"""
scripts/bulk_import_celebrities.py
Automated high-throughput Wikipedia PageImages importer for Hamshakal Finder.
Fetches high-resolution reference portraits, runs SCRFD face validation,
extracts ArcFace embeddings, and registers celebrities in dataset/metadata/celebrities.json.

Usage:
  python scripts/bulk_import_celebrities.py
"""

import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scripts.add_celebrity import add_celebrity

TEMP_DIR = PROJECT_ROOT / "temp_downloads"
TEMP_DIR.mkdir(exist_ok=True)

# Curated list of prominent celebrities to index
TARGET_CELEBRITIES = [
    {"key": "ranbir_kapoor", "title": "Ranbir_Kapoor", "name": "Ranbir Kapoor", "gender": "male", "tags": "bollywood,actor"},
    {"key": "priyanka_chopra", "title": "Priyanka_Chopra", "name": "Priyanka Chopra", "gender": "female", "tags": "bollywood,hollywood,actress"},
    {"key": "katrina_kaif", "title": "Katrina_Kaif", "name": "Katrina Kaif", "gender": "female", "tags": "bollywood,actress"},
    {"key": "ranveer_singh", "title": "Ranveer_Singh", "name": "Ranveer Singh", "gender": "male", "tags": "bollywood,actor"},
    {"key": "kareena_kapoor", "title": "Kareena_Kapoor", "name": "Kareena Kapoor", "gender": "female", "tags": "bollywood,actress"},
    {"key": "amitabh_bachchan", "title": "Amitabh_Bachchan", "name": "Amitabh Bachchan", "gender": "male", "tags": "bollywood,actor"},
    {"key": "aamir_khan", "title": "Aamir_Khan", "name": "Aamir Khan", "gender": "male", "tags": "bollywood,actor"},
    {"key": "leonardo_dicaprio", "title": "Leonardo_DiCaprio", "name": "Leonardo DiCaprio", "gender": "male", "tags": "hollywood,actor"},
    {"key": "ryan_gosling", "title": "Ryan_Gosling", "name": "Ryan Gosling", "gender": "male", "tags": "hollywood,actor"},
    {"key": "emma_watson", "title": "Emma_Watson", "name": "Emma Watson", "gender": "female", "tags": "hollywood,actress"},
    {"key": "zendaya", "title": "Zendaya", "name": "Zendaya", "gender": "female", "tags": "hollywood,actress"}
]


def fetch_wikipedia_image_url(page_title: str) -> Optional[str]:
    """Queries Wikipedia API for highest resolution thumbnail."""
    headers = {"User-Agent": "HamshakalFinderBot/2.0 (education; https://github.com)"}
    endpoint = f"https://en.wikipedia.org/w/api.php?action=query&titles={page_title}&prop=pageimages&format=json&pithumbsize=600"
    req = urllib.request.Request(endpoint, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        pages = data.get("query", {}).get("pages", {})
        for _, page in pages.items():
            thumb = page.get("thumbnail", {}).get("source")
            if thumb:
                return thumb
    except Exception as e:
        print(f"[Warning] Failed to query Wikipedia API for {page_title}: {e}")
    return None


def download_file(url: str, dest_path: Path) -> bool:
    """Downloads an image file with custom User-Agent."""
    headers = {"User-Agent": "HamshakalFinderBot/2.0 (education; https://github.com)"}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            with open(dest_path, "wb") as f:
                f.write(resp.read())
        return True
    except Exception as e:
        print(f"[Warning] Download error for {url}: {e}")
        return False


def main():
    print("\n" + "=" * 65)
    print(" HAMSHAKAL FINDER — BULK CELEBRITY IMPORTER (WIKIPEDIA API)")
    print("=" * 65)
    print(f"Target roster: {len(TARGET_CELEBRITIES)} personalities to evaluate & index.\n")

    metadata_path = PROJECT_ROOT / "dataset" / "metadata" / "celebrities.json"
    existing_keys = set()
    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                existing_keys = set(json.load(f).get("celebrities", {}).keys())
        except Exception:
            pass

    added = 0
    skipped = 0
    failed = 0

    for celeb in TARGET_CELEBRITIES:
        key = celeb["key"]
        name = celeb["name"]
        emb_file = PROJECT_ROOT / "embedding" / f"{key}.npy"

        if key in existing_keys and emb_file.exists():
            print(f"[Skip] '{name}' (key: {key}) is already indexed. Skipping.")
            skipped += 1
            continue

        print(f"\n[*] Fetching Wikipedia portrait for '{name}'...")
        img_url = fetch_wikipedia_image_url(celeb["title"])
        if not img_url:
            print(f"[Fail] Could not locate Wikipedia portrait for '{name}'.")
            failed += 1
            continue

        temp_img = TEMP_DIR / f"{key}.jpg"
        if not download_file(img_url, temp_img):
            print(f"[Fail] Could not download image for '{name}'.")
            failed += 1
            continue

        # Run face validation and indexing
        success = add_celebrity(
            key=key,
            name=name,
            image_src=str(temp_img),
            gender=celeb["gender"],
            tags=celeb["tags"],
            use_nested_folder=False
        )

        if success:
            added += 1
        else:
            failed += 1

        if temp_img.exists():
            try:
                temp_img.unlink()
            except Exception:
                pass

        time.sleep(0.5)  # Polite request spacing

    try:
        TEMP_DIR.rmdir()
    except Exception:
        pass

    print("\n" + "=" * 65)
    print(" BULK IMPORT SUMMARY")
    print("=" * 65)
    print(f"Successfully added & indexed : {added}")
    print(f"Already existed (skipped)    : {skipped}")
    print(f"Failed / Unresolved          : {failed}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
