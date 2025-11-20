"""
cornMoving.py

Usage:
  - Configure src_root and dst_dir (absolute or relative to project).
  - Set selected_base_range to (1,160) to limit genotypes.
  - Set dry_run = True to preview actions without copying.
  - Run: python utils/cornMoving.py
"""

import os
import re
import shutil
import random
import csv
from pathlib import Path
from typing import Tuple, Optional, List

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def extract_genotype_and_individual(folder_name: str) -> Optional[Tuple[int, int]]:
    """
    Try to extract a 3-digit genotype (001..608) and an individual E# from a folder name.
    Returns (genotype_int, individual_int) or None if not found.
    """
    # find first 3-digit number (e.g. 001 or 305)
    m_g = re.search(r"(\d{3})", folder_name)
    m_e = re.search(r"[Ee] *[_-]?\s*(\d+)", folder_name) or re.search(r"[Ee](\d+)", folder_name)
    if not m_g or not m_e:
        return None
    try:
        g = int(m_g.group(1))
        e = int(m_e.group(1))
        return g, e
    except ValueError:
        return None


def map_genotype_and_individual(g: int, e: int) -> Tuple[int, int]:
    """
    Map the raw genotype (1..608) to the 'lower value' genotype and map individual numbers:
      - if g <= 304: base = g, new_individual = e (1 or 2)
      - if g >  304: base = g - 304, new_individual = 3 if e==1 else 4 if e==2
    """
    if g <= 304:
        base = g
        new_e = e
    else:
        base = g - 304
        new_e = 3 if e == 1 else 4 if e == 2 else e
    return base, new_e


def is_image_file(fname: str) -> bool:
    return Path(fname).suffix.lower() in IMAGE_EXTS


def pick_random_image(files: List[str], seed: Optional[int] = None) -> Optional[str]:
    """Pick one random file from list excluding ones with 'tag' in the name (case-insensitive)."""
    candidates = [f for f in files if is_image_file(f) and "tag" not in f.lower()]
    if not candidates:
        return None
    if seed is not None:
        random.seed(seed)
    return random.choice(sorted(candidates))


def collect_and_copy(
    src_root: str,
    dst_dir: str,
    selected_base_range: Tuple[int, int] = (1, 160),
    dry_run: bool = True,
    seed: Optional[int] = 42,
    save_mapping_csv: Optional[str] = "./results/copy_mapping.csv",
) -> None:
    src_root = Path(src_root)
    dst_dir = Path(dst_dir)
    if not src_root.exists():
        raise FileNotFoundError(f"Source root not found: {src_root}")

    if not dry_run:
        dst_dir.mkdir(parents=True, exist_ok=True)

    # sort collection date folders by name
    collections = sorted([d for d in src_root.iterdir() if d.is_dir()], key=lambda p: p.name)
    mapping_rows = []
    collection_index = 0

    for coll_folder in collections:
        collection_index += 1  # 1-based index as requested
        # inside collection folder: genotype folders
        genotype_folders = sorted([d for d in coll_folder.iterdir() if d.is_dir()], key=lambda p: p.name)
        for gf in genotype_folders:
            parsed = extract_genotype_and_individual(gf.name)
            if parsed is None:
                # skip folders that do not match expected pattern
                continue
            raw_g, raw_e = parsed
            base_g, new_e = map_genotype_and_individual(raw_g, raw_e)

            # Only keep specified range
            if not (selected_base_range[0] <= base_g <= selected_base_range[1]):
                continue

            # list image files
            files = [f.name for f in gf.iterdir() if f.is_file() and is_image_file(f.name)]
            if not files:
                mapping_rows.append({
                    "collection_index": collection_index,
                    "collection_folder": str(coll_folder),
                    "genotype_folder": str(gf),
                    "raw_genotype": raw_g,
                    "raw_individual": raw_e,
                    "mapped_genotype": base_g,
                    "mapped_individual": new_e,
                    "picked_file": "",
                    "dst_path": "",
                    "status": "no_images"
                })
                continue

            pick = pick_random_image(files, seed=seed)
            if pick is None:
                mapping_rows.append({
                    "collection_index": collection_index,
                    "collection_folder": str(coll_folder),
                    "genotype_folder": str(gf),
                    "raw_genotype": raw_g,
                    "raw_individual": raw_e,
                    "mapped_genotype": base_g,
                    "mapped_individual": new_e,
                    "picked_file": "",
                    "dst_path": "",
                    "status": "no_candidate"
                })
                continue

            src_path = gf / pick
            # use only numbers in the destination name (no original filename, no 'E')
            ext = Path(pick).suffix
            dst_fname = f"{collection_index}_{base_g}_{new_e}{ext}"
            dst_path_planned = dst_dir / dst_fname

            # If actually copying, resolve collisions and record the final destination
            dst_path_to_record = str(dst_path_planned)
            status = "planned"
            if not dry_run:
                dst_dir.mkdir(parents=True, exist_ok=True)
                final_dst = dst_path_planned
                counter = 1
                while final_dst.exists():
                    final_dst = dst_dir / f"{collection_index}_{base_g}_{new_e}_{counter}{ext}"
                    counter += 1
                shutil.copy2(src_path, final_dst)
                dst_path_to_record = str(final_dst)
                status = "copied"

            mapping_rows.append({
                "collection_index": collection_index,
                "collection_folder": str(coll_folder),
                "genotype_folder": str(gf),
                "raw_genotype": raw_g,
                "raw_individual": raw_e,
                "mapped_genotype": base_g,
                "mapped_individual": new_e,
                "picked_file": str(src_path),
                "dst_path": dst_path_to_record,
                "status": status
            })

    # save mapping CSV
    if save_mapping_csv:
        Path(save_mapping_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(save_mapping_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=[
                "collection_index", "collection_folder", "genotype_folder",
                "raw_genotype", "raw_individual", "mapped_genotype", "mapped_individual",
                "picked_file", "dst_path", "status"
            ])
            writer.writeheader()
            for r in mapping_rows:
                writer.writerow(r)

    # summary
    total_planned = sum(1 for r in mapping_rows if r["status"] in ("planned", "copied"))
    print(f"Collections scanned: {len(collections)}")
    print(f"Total sampled images (planned/copied): {total_planned}")
    print(f"Mapping CSV written to: {save_mapping_csv}")
    if dry_run:
        print("Dry run enabled — no files were copied. Review the mapping CSV and set dry_run=False to execute.")


if __name__ == "__main__":
    # Example configuration — adjust paths as needed.
    SRC_ROOT = "E:\\maize_tuinstra_download\\Processed"  # root that contains collection-date folders
    DST_DIR = "./../data/corn/images"  # single folder where images will be copied
    SELECTED_BASE_RANGE = (1, 160)  # genotypes 1..160 (and their 305..465 counterparts)
    DRY_RUN = False  # change to False to perform actual copying
    RANDOM_SEED = 42
    MAPPING_CSV = "./results/copy_mapping_crossgenotype.csv"

    collect_and_copy(
        src_root=SRC_ROOT,
        dst_dir=DST_DIR,
        selected_base_range=SELECTED_BASE_RANGE,
        dry_run=DRY_RUN,
        seed=RANDOM_SEED,
        save_mapping_csv=MAPPING_CSV,
    )