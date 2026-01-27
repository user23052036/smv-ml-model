#!/usr/bin/env python3
"""
find_image_duplicates.py

Find exact and perceptual duplicates in an image dataset tree.
Safe default: generates a CSV report and moves duplicates to ./duplicates/
(keeps one representative per duplicate group).

Usage examples:
  pip install Pillow imagehash tqdm
  python find_image_duplicates.py --root dataset --report-only
  python find_image_duplicates.py --root dataset --move --workers 8 --phash-threshold 6
"""

import os
import argparse
import hashlib
from PIL import Image, UnidentifiedImageError
import imagehash
from tqdm import tqdm
from collections import defaultdict
import csv
import shutil
from multiprocessing import Pool, cpu_count
import numpy as np

# ---------- Utility: convert ImageHash -> int (safe) ----------
def imagehash_to_int(ph):
    """
    Convert imagehash.ImageHash -> Python int deterministically.
    ph.hash is a 2D numpy bool array; we flatten row-major and pack bits
    so hamming distance on ints matches ImageHash behavior.
    """
    if ph is None:
        return None
    bits = ph.hash.flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(bool(b))
    return value

# ---------- Helpers ----------
def iter_image_files(root, exts=(".jpg", ".jpeg", ".png", ".bmp", ".gif")):
    lower_exts = tuple(e.lower() for e in exts)
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(lower_exts):
                yield os.path.join(dirpath, f)

def file_hash(path, algo="md5", blocksize=65536):
    try:
        h = hashlib.new(algo)
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(blocksize), b""):
                h.update(block)
        return h.hexdigest()
    except Exception:
        return None

def phash_image(path, hash_size=8):
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            return imagehash.phash(img, hash_size=hash_size)  # ImageHash object
    except UnidentifiedImageError:
        return None
    except Exception:
        # corrupted file or unsupported format
        return None

# Worker for parallel hashing
def worker_hash(args):
    """
    args: (path, do_filehash, do_phash, hash_algo, phash_size)
    returns: {"path": path, "file_hash": ..., "phash": int or None}
    """
    path, do_filehash, do_phash, hash_algo, phash_size = args
    result = {"path": path, "file_hash": None, "phash": None}
    if do_filehash:
        result["file_hash"] = file_hash(path, algo=hash_algo)
    if do_phash:
        ph = phash_image(path, hash_size=phash_size)
        result["phash"] = imagehash_to_int(ph)
    return result

# Hamming distance between integer phashes (assumes ints)
def hamming_int(a, b):
    return (a ^ b).bit_count()

# ---------- Duplicate detection ----------
def find_duplicates(root, report_only=True, move=False, out_dir="duplicates",
                    phash_threshold=6, prefix_bits=16, workers=4, hash_algo="md5",
                    phash_size=8):
    files = list(iter_image_files(root))
    if not files:
        print("No image files found under", root)
        return

    print(f"Found {len(files)} image files. Hashing (file hash + phash)...")

    do_filehash = True
    do_phash = True
    pool_args = [(p, do_filehash, do_phash, hash_algo, phash_size) for p in files]

    workers = max(1, min(workers, cpu_count()))
    results = []
    with Pool(workers) as pool:
        for res in tqdm(pool.imap_unordered(worker_hash, pool_args), total=len(pool_args)):
            results.append(res)

    # Group by exact file hash
    filehash_map = defaultdict(list)
    for r in results:
        fh = r.get("file_hash")
        if fh:
            filehash_map[fh].append(r["path"])

    exact_groups = [v for v in filehash_map.values() if len(v) > 1]
    print(f"Exact duplicate groups: {len(exact_groups)}")

    # Perceptual duplicates (phash ints)
    phash_map = {}
    for r in results:
        ph = r.get("phash")
        if ph is not None:
            phash_map[r["path"]] = ph

    if not phash_map:
        print("No perceptual hashes computed (phash map empty).")
        phash_groups = []
    else:
        # Bucketing by prefix to reduce comparisons
        bitlength = max((phash.bit_length() for phash in phash_map.values()), default=phash_size*phash_size)
        shift = max(0, bitlength - prefix_bits)
        buckets = defaultdict(list)
        for path, ph in phash_map.items():
            bucket = (ph >> shift) if shift > 0 else ph
            buckets[bucket].append((path, ph))

        # Union-Find to collect groups
        parent = {}
        def find(x):
            parent.setdefault(x, x)
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        exact_paths = set(p for grp in exact_groups for p in grp)

        for bucket_items in buckets.values():
            n = len(bucket_items)
            if n <= 1:
                continue
            for i in range(n):
                pi, phi = bucket_items[i]
                for j in range(i + 1, n):
                    pj, phj = bucket_items[j]
                    # Skip if identical file hash already handled
                    if pi in exact_paths or pj in exact_paths:
                        continue
                    dist = hamming_int(phi, phj)
                    if dist <= phash_threshold:
                        union(pi, pj)

        groups = defaultdict(list)
        for p in phash_map.keys():
            if p in parent:
                groups[find(p)].append(p)
        phash_groups = [v for v in groups.values() if len(v) > 1]

    print(f"Perceptual duplicate groups (threshold={phash_threshold}): {len(phash_groups)}")

    # Prepare CSV report rows
    rows = []
    gid = 0
    # exact groups
    for g in exact_groups:
        gid += 1
        rep = sorted(g)[0]
        for p in g:
            rows.append({"group_id": gid, "type": "exact", "representative": rep, "path": p})
    # perceptual groups
    for g in phash_groups:
        gid += 1
        rep = sorted(g)[0]
        for p in g:
            rows.append({"group_id": gid, "type": "perceptual", "representative": rep, "path": p})

    os.makedirs(out_dir, exist_ok=True)
    report_csv = os.path.join(out_dir, "duplicate_report.csv")
    with open(report_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["group_id", "type", "representative", "path"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print("Report written to:", report_csv)
    print(f"Total duplicate entries (rows): {len(rows)}")

    if report_only:
        print("Report-only mode: not moving anything. Inspect the CSV and decide.")
        return

    # MOVE duplicates (keep representative in place)
    moved = 0
    for entry in rows:
        group = entry["group_id"]
        rep = entry["representative"]
        p = entry["path"]
        if os.path.abspath(p) == os.path.abspath(rep):
            continue
        tgt_dir = os.path.join(out_dir, f"group_{group}")
        os.makedirs(tgt_dir, exist_ok=True)
        try:
            shutil.move(p, os.path.join(tgt_dir, os.path.basename(p)))
            moved += 1
        except Exception as e:
            print(f"[WARN] Failed to move {p}: {e}")

    print(f"Moved {moved} duplicate files into: {os.path.abspath(out_dir)}")
    print("IMPORTANT: Review them before deletion. Re-run training with cleaned dataset.")

# ---------- CLI ----------
def parse_args_and_run():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="Dataset root (will walk recursively).")
    p.add_argument("--report-only", action="store_true", help="Only produce CSV report (don't move).")
    p.add_argument("--move", action="store_true", help="Move detected duplicates into ./duplicates/. This is safer than delete.")
    p.add_argument("--out-dir", default="duplicates", help="Directory to store duplicates and CSV.")
    p.add_argument("--phash-threshold", type=int, default=6, help="Hamming distance threshold for phash.")
    p.add_argument("--prefix-bits", type=int, default=16, help="Bucket prefix bits to reduce comparisons.")
    p.add_argument("--workers", type=int, default=min(4, cpu_count()), help="Parallel workers for hashing.")
    p.add_argument("--hash-algo", default="md5", help="File hash algorithm (md5/sha1...).")
    p.add_argument("--phash-size", type=int, default=8, help="phash hash_size (8 => 8x8 => 64 bits).")
    args = p.parse_args()

    # Default safe behaviour
    if not args.report_only and not args.move:
        print("No action specified: defaulting to --report-only (safe). Use --move to relocate duplicates.")
        args.report_only = True

    find_duplicates(
        args.root,
        report_only=args.report_only,
        move=args.move,
        out_dir=args.out_dir,
        phash_threshold=args.phash_threshold,
        prefix_bits=args.prefix_bits,
        workers=args.workers,
        hash_algo=args.hash_algo,
        phash_size=args.phash_size,
    )

if __name__ == "__main__":
    parse_args_and_run()
