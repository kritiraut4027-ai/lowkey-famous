#!/usr/bin/env python3
"""
scripts/download_sample_celebrities.py
Automated utility to download a curated batch of public-domain/Creative Commons
celebrity portraits from Wikimedia Commons and index them into Hamshakal Finder.

Usage:
  python scripts/download_sample_celebrities.py
"""

import os
import sys
import time
from pathlib import Path
import urllib.request

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

# Curated list of high-quality, front-facing celebrity portraits from Wikimedia Commons
SAMPLE_CELEBRITIES = [
    {
        "key": "hrithik_roshan",
        "name": "Hrithik Roshan",
        "gender": "male",
        "tags": "bollywood,actor",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/cc/Hrithik_Roshan_2014.jpg/480px-Hrithik_Roshan_2014.jpg"
    },
    {
        "key": "priyanka_chopra",
        "name": "Priyanka Chopra",
        "gender": "female",
        "tags": "bollywood,hollywood,actress",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6c/Priyanka-chopra-gesf-2018-7565.jpg/480px-Priyanka-chopra-gesf-2018-7565.jpg"
    },
    {
        "key": "ranbir_kapoor",
        "name": "Ranbir Kapoor",
        "gender": "male",
        "tags": "bollywood,actor",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/01/Ranbir_Kapoor_promoting_Brahmastra.jpg/480px-Ranbir_Kapoor_promoting_Brahmastra.jpg"
    },
    {
        "key": "katrina_kaif",
        "name": "Katrina Kaif",
        "gender": "female",
        "tags": "bollywood,actress",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8b/Katrina_Kaif_promoting_Bharat_in_2019.jpg/480px-Katrina_Kaif_promoting_Bharat_in_2019.jpg"
    },
    {
        "key": "leonardo_dicaprio",
        "name": "Leonardo DiCaprio",
        "gender": "male",
        "tags": "hollywood,actor",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/25/Leonardo_DiCaprio_2014.jpg/480px-Leonardo_DiCaprio_2014.jpg"
    }
]


def download_image(url: str, dest_path: Path) -> bool:
    """Downloads an image file with standard browser user-agent header."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
            with open(dest_path, "wb") as f:
                f.write(data)
        return True
    except Exception as e:
        print(f"[Warning] Failed to download {url}: {e}")
        return False


def main():
    print("\n" + "=" * 65)
    print(" HAMSHAKAL FINDER — EXPAND DATASET WITH SAMPLE CELEBRITIES")
    print("=" * 65)
    print(f"Discovered {len(SAMPLE_CELEBRITIES)} sample portraits to fetch and index.\n")

    added = 0
    for celeb in SAMPLE_CELEBRITIES:
        temp_img = TEMP_DIR / f"{celeb['key']}.jpg"
        print(f"[*] Downloading photo for {celeb['name']}...")
        if not download_image(celeb["url"], temp_img):
            continue

        # Add using add_celebrity script
        success = add_celebrity(
            key=celeb["key"],
            name=celeb["name"],
            image_src=str(temp_img),
            gender=celeb["gender"],
            tags=celeb["tags"],
            use_nested_folder=False
        )
        if success:
            added += 1
            if temp_img.exists():
                try:
                    temp_img.unlink()
                except Exception:
                    pass

    # Clean up temp dir
    try:
        TEMP_DIR.rmdir()
    except Exception:
        pass

    print("\n" + "=" * 65)
    print(f"🎉 DATASET EXPANSION COMPLETE: Successfully indexed {added} new celebrities!")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
