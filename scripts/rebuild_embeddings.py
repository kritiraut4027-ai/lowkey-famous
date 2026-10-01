#!/usr/bin/env python3
"""
scripts/rebuild_embeddings.py
Clean complete rebuild utility for Hamshakal Finder celebrity embeddings.
Re-extracts ArcFace embeddings for all discovered images and updates the metadata registry.

Usage:
  python scripts/rebuild_embeddings.py [--yes]
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_embeddings import run_embedding_generation


def main():
    parser = argparse.ArgumentParser(description="Rebuild all celebrity embeddings from scratch")
    parser.add_argument("--yes", "-y", action="store_true", help="Confirm rebuild without interactive prompt")
    args = parser.parse_args()

    if not args.yes:
        print("\n⚠️ WARNING: This will re-extract embeddings for ALL celebrity photos in the dataset.")
        print("Existing .npy embedding files will be overwritten with freshly computed vectors.")
        confirm = input("Are you sure you want to rebuild all embeddings? [y/N]: ").strip().lower()
        if confirm not in ["y", "yes"]:
            print("Operation aborted by user.")
            sys.exit(0)

    print("\nStarting complete rebuild of celebrity embeddings...")
    report = run_embedding_generation(mode="all", force=True)
    if report.get("failed", 0) > 0:
        print(f"\nCompleted with {report['failed']} failures. See validation_reports/ for details.")
    else:
        print("\n🎉 Rebuild completed successfully with 0 failures!")


if __name__ == "__main__":
    main()
