#!/usr/bin/env python3
"""
keep_best_train_duplicates.py

Keeps one best image per duplicate group (train/) and moves the rest to duplicates/train_groups/.
Safe by default: add --dry-run to preview actions without moving files.
"""

import csv
import os
import shutil
import argparse
from PIL import Image

def safe_move(src, dst):
    """Move src -> dst; if dst exists, create a unique name."""
    base, ext = os.path.splitext(dst)
    i = 1
    target = dst
    while os.path.exists(target):
        target = f"{base}__dup{i}{ext}"
        i += 1
    shutil.move(src, target)
    return target

def score_image(path):
    """Primary score: resolution (w*h). Fallback: file size. Returns int score."""
    try:
        with Image.open(path) as im:
            w, h = im.size
            return int(w) * int(h)
    except Exception:
        try:
            return os.path.getsize(path)
        except Exception:
            return 0

def main(report_path, out_base, dry_run):
    if not os.path.exists(report_path):
        raise SystemExit(f"Report not found: {report_path}")

    os.makedirs(out_base, exist_ok=True)

    # Load groups from CSV
    groups = {}
    with open(report_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gid = row.get("group_id")
            path = row.get("path", "").replace("\\", "/").strip()
            if not path:
                continue
            groups.setdefault(gid, []).append(path)

    groups_processed = 0
    files_moved = 0
    missing_paths = 0

    for gid, paths in groups.items():
        # only care about TRAIN duplicates
        train_paths = [p for p in paths if "/train/" in p]

        if len(train_paths) <= 1:
            continue  # nothing to deduplicate

        # filter out non-existing paths but keep note
        existing = []
        for p in train_paths:
            if os.path.exists(p):
                existing.append(p)
            else:
                missing_paths += 1

        if len(existing) <= 1:
            continue

        # choose best representative (highest resolution, tie-break by filesize)
        best_path = None
        best_score = -1
        best_size = -1
        for p in existing:
            s = score_image(p)
            sz = os.path.getsize(p) if os.path.exists(p) else 0
            if (s > best_score) or (s == best_score and sz > best_size):
                best_score = s
                best_size = sz
                best_path = p

        if best_path is None:
            continue

        groups_processed += 1

        # move other images
        target_dir = os.path.join(out_base, f"group_{gid}")
        if not dry_run:
            os.makedirs(target_dir, exist_ok=True)

        for p in existing:
            try:
                if os.path.abspath(p) == os.path.abspath(best_path):
                    continue
                dst = os.path.join(target_dir, os.path.basename(p))
                if dry_run:
                    print(f"[DRY] Would move: {p} -> {dst}")
                else:
                    moved_to = safe_move(p, dst)
                    files_moved += 1
                    print(f"Moved: {p} -> {moved_to}")
            except Exception as e:
                print(f"[WARN] Failed to move {p}: {e}")

    print("\nSummary:")
    print("  Groups processed:", groups_processed)
    print("  Files moved:", files_moved)
    print("  Missing paths in report (skipped):", missing_paths)
    print("Duplicates moved to:", os.path.abspath(out_base))
    if dry_run:
        print("DRY RUN: no files were moved.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--report", default="duplicates/duplicate_report.csv", help="path to duplicate_report.csv")
    p.add_argument("--out", default="duplicates/train_groups", help="output directory for moved duplicates")
    p.add_argument("--dry-run", action="store_true", help="show actions but do not move files")
    args = p.parse_args()
    main(args.report, args.out, args.dry_run)
