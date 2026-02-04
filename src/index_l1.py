#!/usr/bin/env python3
"""
L1: Pattern Library Indexer (Google Drive mirror -> SQLite + CSV report)

What it does
- Scans Patterns/data/** (local folder mirror)
- Builds an inventory of assets (images/pdfs/others)
- Groups assets into "patterns" using:
  A) Folder unit: any subfolder inside a category folder is one pattern
  B) File group: files directly inside a category folder
     - explicit back naming: b_<frontname>.jpg
     - suffix naming: *_front.jpg / *_back.jpg (or *_f / *_b)
     - IMG_### pairing: back is next number after front (IMG_123 -> IMG_124)
- Writes:
  - SQLite DB: pattern_library.sqlite
  - CSV report: pattern_inventory_report.csv (now includes counts + has_pdf)

Usage
  python3 index_l1.py --root "/path/to/Patterns/data" --out "./out"

Notes
- If your folder is in Google Drive CloudStorage, ensure files are available offline if you see read issues.
"""

import argparse
import csv
import hashlib
import os
import re
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
PDF_EXTS = {".pdf"}

IMG_RE = re.compile(r"^IMG_(\d+)$", re.IGNORECASE)

# Common suffixes you might use (we support both)
FRONT_SUFFIX_RE = re.compile(r"(?:^|[_\-])(front|f)$", re.IGNORECASE)
BACK_SUFFIX_RE = re.compile(r"(?:^|[_\-])(back|b)$", re.IGNORECASE)


@dataclass
class Asset:
    asset_id: str
    asset_type: str  # front_image, back_image, pdf, other_image, unknown_image, other
    file_name: str
    file_extension: str
    category: str
    folder_path: str        # relative to root (Patterns/data)
    full_path: str          # absolute path
    file_size_bytes: int
    modified_at: str        # ISO string


@dataclass
class PatternRec:
    pattern_id: str
    category: str
    pattern_group_type: str     # folder_unit or file_group
    pattern_group_key: str      # folder path or grouping key
    front_asset_id: Optional[str]
    back_asset_id: Optional[str]
    confidence: float
    notes: str
    created_at: str
    updated_at: str


def iso_from_mtime(mtime: float) -> str:
    return datetime.fromtimestamp(mtime).isoformat(timespec="seconds")


def stable_id_from_relpath(relpath: str) -> str:
    h = hashlib.sha1(relpath.encode("utf-8")).hexdigest()[:16]
    return h


def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w\-]+", "_", s)
    s = re.sub(r"__+", "_", s)
    s = s.strip("_")
    return s or "unnamed"


def ensure_tables(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS pattern_assets (
        asset_id TEXT PRIMARY KEY,
        asset_type TEXT,
        file_name TEXT,
        file_extension TEXT,
        category TEXT,
        folder_path TEXT,
        full_path TEXT,
        file_size_bytes INTEGER,
        modified_at TEXT
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS patterns (
        pattern_id TEXT PRIMARY KEY,
        category TEXT,
        pattern_group_type TEXT,
        pattern_group_key TEXT,
        front_asset_id TEXT,
        back_asset_id TEXT,
        confidence REAL,
        notes TEXT,
        created_at TEXT,
        updated_at TEXT
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS pattern_asset_links (
        pattern_id TEXT,
        asset_id TEXT,
        role TEXT,
        PRIMARY KEY (pattern_id, asset_id)
    )
    """)
    conn.commit()


def upsert_assets(conn: sqlite3.Connection, assets: List[Asset]) -> None:
    cur = conn.cursor()
    cur.executemany("""
    INSERT OR REPLACE INTO pattern_assets (
        asset_id, asset_type, file_name, file_extension, category,
        folder_path, full_path, file_size_bytes, modified_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        (a.asset_id, a.asset_type, a.file_name, a.file_extension, a.category,
         a.folder_path, a.full_path, a.file_size_bytes, a.modified_at)
        for a in assets
    ])
    conn.commit()


def upsert_patterns(conn: sqlite3.Connection, patterns: List[PatternRec]) -> None:
    cur = conn.cursor()
    cur.executemany("""
    INSERT OR REPLACE INTO patterns (
        pattern_id, category, pattern_group_type, pattern_group_key,
        front_asset_id, back_asset_id, confidence, notes, created_at, updated_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        (p.pattern_id, p.category, p.pattern_group_type, p.pattern_group_key,
         p.front_asset_id, p.back_asset_id, p.confidence, p.notes, p.created_at, p.updated_at)
        for p in patterns
    ])
    conn.commit()


def upsert_links(conn: sqlite3.Connection, links: List[Tuple[str, str, str]]) -> None:
    cur = conn.cursor()
    cur.executemany("""
    INSERT OR REPLACE INTO pattern_asset_links (pattern_id, asset_id, role)
    VALUES (?, ?, ?)
    """, links)
    conn.commit()


def classify_asset_type(path: Path) -> str:
    ext = path.suffix.lower()
    stem = path.stem

    if ext in PDF_EXTS:
        return "pdf"

    if ext in IMG_EXTS:
        # Explicit conventions
        if stem.lower().startswith("b_"):
            return "back_image"
        if BACK_SUFFIX_RE.search(stem):
            return "back_image"
        if FRONT_SUFFIX_RE.search(stem):
            return "front_image"
        # Unknown until grouping
        return "unknown_image"

    return "other"


def scan_assets(root: Path) -> List[Asset]:
    assets: List[Asset] = []
    root = root.resolve()

    for category_dir in sorted([p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")]):
        category = category_dir.name

        for path in category_dir.rglob("*"):
            if path.is_dir():
                continue
            if path.name.startswith("."):
                continue

            rel = path.relative_to(root).as_posix()
            folder_path = str(Path(rel).parent).replace("\\", "/")
            asset_id = stable_id_from_relpath(rel)

            try:
                stat = path.stat()
                size = stat.st_size
                mtime = iso_from_mtime(stat.st_mtime)
            except OSError:
                size = 0
                mtime = ""

            assets.append(Asset(
                asset_id=asset_id,
                asset_type=classify_asset_type(path),
                file_name=path.name,
                file_extension=path.suffix.lower(),
                category=category,
                folder_path=folder_path,
                full_path=str(path.resolve()),
                file_size_bytes=size,
                modified_at=mtime
            ))

    return assets


def assets_by_category(assets: List[Asset]) -> Dict[str, List[Asset]]:
    out: Dict[str, List[Asset]] = {}
    for a in assets:
        out.setdefault(a.category, []).append(a)
    return out


def group_folder_units(cat_assets: List[Asset], category: str) -> Dict[str, List[Asset]]:
    """
    Returns dict: folder_rel_path (e.g., 'Dress/Simplicity_123') -> assets in that folder subtree
    Only for subfolders under category.
    """
    groups: Dict[str, List[Asset]] = {}
    for a in cat_assets:
        if a.folder_path == category:
            continue  # direct files handled elsewhere

        parts = a.folder_path.split("/")
        if len(parts) >= 2 and parts[0] == category:
            key = "/".join(parts[:2])  # category/<top_subfolder>
            groups.setdefault(key, []).append(a)

    return groups


def group_direct_files(cat_assets: List[Asset], category: str) -> List[List[Asset]]:
    """
    For assets directly inside the category folder, create groups (pattern units).
    Strategy:
    - Pair explicit b_<front>.jpg with <front>.jpg where possible
    - Pair *_front with *_back where possible
    - Pair IMG_### with next IMG_### for back (front=IMG_n, back=IMG_{n+1})
    - Remaining images are grouped by base stem
    - PDFs/others remain singletons (safe, but uncommon in your setup)
    """
    direct = [a for a in cat_assets if a.folder_path == category]
    images = [a for a in direct if a.file_extension in IMG_EXTS]
    pdfs = [a for a in direct if a.file_extension in PDF_EXTS]
    others = [a for a in direct if a.file_extension not in IMG_EXTS and a.file_extension not in PDF_EXTS]

    by_stem: Dict[str, Asset] = {Path(a.file_name).stem.lower(): a for a in images}
    used: set[str] = set()
    groups: List[List[Asset]] = []

    # 1) Pair b_<front> with <front>
    for stem, back_asset in list(by_stem.items()):
        if stem.startswith("b_"):
            front_stem = stem[2:]
            if front_stem in by_stem and back_asset.asset_id not in used and by_stem[front_stem].asset_id not in used:
                groups.append([by_stem[front_stem], back_asset])
                used.add(back_asset.asset_id)
                used.add(by_stem[front_stem].asset_id)

    # helper: strip suffix
    def strip_suffix(stem: str) -> Tuple[str, Optional[str]]:
        if BACK_SUFFIX_RE.search(stem):
            base = BACK_SUFFIX_RE.sub("", stem)
            return base.strip("_-"), "back"
        if FRONT_SUFFIX_RE.search(stem):
            base = FRONT_SUFFIX_RE.sub("", stem)
            return base.strip("_-"), "front"
        return stem, None

    # 2) Pair *_front with *_back
    suffix_map: Dict[str, Dict[str, Asset]] = {}
    for a in images:
        if a.asset_id in used:
            continue
        stem = Path(a.file_name).stem.lower()
        base, which = strip_suffix(stem)
        if which:
            suffix_map.setdefault(base, {})[which] = a

    for base, d in suffix_map.items():
        if "front" in d and "back" in d:
            if d["front"].asset_id not in used and d["back"].asset_id not in used:
                groups.append([d["front"], d["back"]])
                used.add(d["front"].asset_id)
                used.add(d["back"].asset_id)

    # 3) Pair IMG_### with IMG_### + 1
    img_assets: Dict[int, Asset] = {}
    for a in images:
        if a.asset_id in used:
            continue
        stem = Path(a.file_name).stem
        m = IMG_RE.match(stem)
        if m:
            img_assets[int(m.group(1))] = a

    for n in sorted(img_assets.keys()):
        if img_assets[n].asset_id in used:
            continue
        if (n + 1) in img_assets and img_assets[n + 1].asset_id not in used:
            groups.append([img_assets[n], img_assets[n + 1]])
            used.add(img_assets[n].asset_id)
            used.add(img_assets[n + 1].asset_id)

    # 4) Remaining images grouped by base stem
    remaining_imgs = [a for a in images if a.asset_id not in used]
    base_groups: Dict[str, List[Asset]] = {}
    for a in remaining_imgs:
        stem = Path(a.file_name).stem.lower()
        if stem.startswith("b_"):
            stem = stem[2:]
        base, _ = strip_suffix(stem)
        base_groups.setdefault(base, []).append(a)

    for _, lst in base_groups.items():
        groups.append(lst)

    # PDFs/others singletons (should be rare for your structure)
    for a in pdfs:
        groups.append([a])
    for a in others:
        groups.append([a])

    return groups


def pick_front_back(assets: List[Asset]) -> Tuple[Optional[str], Optional[str], float, str, List[Tuple[str, str, str]]]:
    """
    Choose front and back among assets in a group.
    Updated behavior:
    - If ANY image exists, we ALWAYS pick a front (best-effort).
    - Back may remain missing (allowed).
    """
    imgs = [a for a in assets if a.file_extension in IMG_EXTS]
    pdfs = [a for a in assets if a.file_extension in PDF_EXTS]
    others = [a for a in assets if a.file_extension not in IMG_EXTS and a.file_extension not in PDF_EXTS]

    front_id = None
    back_id = None
    confidence = 0.45
    notes_parts: List[str] = []

    # Preferred: explicit types
    typed_fronts = [a for a in imgs if a.asset_type == "front_image"]
    typed_backs = [a for a in imgs if a.asset_type == "back_image"]
    if typed_fronts:
        front_id = typed_fronts[0].asset_id
        confidence = max(confidence, 0.9)
        notes_parts.append("Front chosen via explicit naming.")
    if typed_backs:
        back_id = typed_backs[0].asset_id
        confidence = max(confidence, 0.9)
        notes_parts.append("Back chosen via explicit naming.")

    # b_ pairing inside group
    if (front_id is None or back_id is None) and imgs:
        by_stem = {Path(a.file_name).stem.lower(): a for a in imgs}
        for stem, a in by_stem.items():
            if stem.startswith("b_"):
                front_stem = stem[2:]
                if front_stem in by_stem:
                    front_id = by_stem[front_stem].asset_id
                    back_id = a.asset_id
                    confidence = max(confidence, 0.95)
                    notes_parts.append("Front/back paired via b_ prefix match.")
                    break

    # IMG sequential inside group
    if (front_id is None or back_id is None) and len(imgs) >= 2:
        def img_sort_key(a: Asset):
            stem = Path(a.file_name).stem
            m = IMG_RE.match(stem)
            if m:
                return (0, int(m.group(1)))
            return (1, a.file_name.lower())

        ordered = sorted(imgs, key=img_sort_key)

        if front_id is None:
            front_id = ordered[0].asset_id
        if back_id is None:
            back_id = ordered[1].asset_id

        confidence = max(confidence, 0.72)
        notes_parts.append("Heuristic front/back from ordering (front then back).")

    # FINAL FALLBACK: ALWAYS pick a front if any image exists
    if imgs and front_id is None:
        # Prefer non-back-labeled image if possible
        non_back = [a for a in imgs if a.asset_type != "back_image" and not Path(a.file_name).stem.lower().startswith("b_")]
        if non_back:
            chosen = sorted(non_back, key=lambda a: a.file_name.lower())[0]
            front_id = chosen.asset_id
            confidence = max(confidence, 0.65)
            notes_parts.append("Fallback: picked a likely front image (non-back-labeled).")
        else:
            chosen = sorted(imgs, key=lambda a: a.file_name.lower())[0]
            front_id = chosen.asset_id
            confidence = max(confidence, 0.60)
            notes_parts.append("Fallback: only back-labeled images found; picked one as front.")

    # Missing back is allowed
    if front_id and not back_id and imgs:
        notes_parts.append("Back missing or not detected.")
        confidence = min(max(confidence, 0.60), 0.85)

    # If no images at all, front remains None (pdf-only patterns etc.)
    if not imgs:
        if pdfs:
            notes_parts.append("No images found (PDF-only pattern group).")
            confidence = max(confidence, 0.55)
        else:
            notes_parts.append("No images found.")
            confidence = max(confidence, 0.50)

    # Build links
    links: List[Tuple[str, str, str]] = []
    for a in assets:
        role = "unknown"
        if a.asset_id == front_id:
            role = "front"
        elif a.asset_id == back_id:
            role = "back"
        elif a.file_extension in PDF_EXTS:
            role = "pdf"
        elif a.file_extension in IMG_EXTS:
            role = "extra"
        else:
            role = "unknown"
        links.append(("", a.asset_id, role))

    notes = " ".join(notes_parts).strip()
    confidence = float(round(confidence, 3))
    return front_id, back_id, confidence, notes, links


def make_pattern_id(category: str, group_type: str, group_key: str, assets: List[Asset]) -> str:
    if group_type == "folder_unit":
        folder_name = group_key.split("/", 1)[1] if "/" in group_key else group_key
        return f"{slugify(category)}__{slugify(folder_name)}"

    img_assets = [a for a in assets if a.file_extension in IMG_EXTS]
    if img_assets:
        stems = [Path(a.file_name).stem for a in img_assets]
        for s in stems:
            m = IMG_RE.match(s)
            if m:
                return f"{slugify(category)}__img_{int(m.group(1)):04d}"

        stem0 = Path(sorted(img_assets, key=lambda a: a.file_name.lower())[0].file_name).stem.lower()
        if stem0.startswith("b_"):
            stem0 = stem0[2:]
        stem0 = FRONT_SUFFIX_RE.sub("", stem0)
        stem0 = BACK_SUFFIX_RE.sub("", stem0)
        stem0 = stem0.strip("_-")
        return f"{slugify(category)}__{slugify(stem0)}"

    h = hashlib.sha1(group_key.encode("utf-8")).hexdigest()[:8]
    return f"{slugify(category)}__group_{h}"


def build_patterns(root: Path, assets: List[Asset]) -> Tuple[List[PatternRec], List[Tuple[str, str, str]]]:
    now = datetime.now().isoformat(timespec="seconds")
    patterns: List[PatternRec] = []
    links_out: List[Tuple[str, str, str]] = []

    cat_map = assets_by_category(assets)

    for category, cat_assets in sorted(cat_map.items()):
        # Folder units
        folder_groups = group_folder_units(cat_assets, category)
        for folder_key, grp_assets in sorted(folder_groups.items(), key=lambda x: x[0].lower()):
            group_type = "folder_unit"
            group_key = folder_key
            pid = make_pattern_id(category, group_type, group_key, grp_assets)
            front_id, back_id, conf, notes, grp_links = pick_front_back(grp_assets)
            links_out.extend([(pid, aid, role) for _, aid, role in grp_links])

            patterns.append(PatternRec(
                pattern_id=pid,
                category=category,
                pattern_group_type=group_type,
                pattern_group_key=group_key,
                front_asset_id=front_id,
                back_asset_id=back_id,
                confidence=conf,
                notes=notes,
                created_at=now,
                updated_at=now
            ))

        # Direct file groups
        direct_groups = group_direct_files(cat_assets, category)
        for grp_assets in direct_groups:
            group_type = "file_group"
            rels = sorted([Path(a.folder_path) / a.file_name for a in grp_assets])
            group_key = f"{category}/__file_group__/" + "|".join([r.as_posix() for r in rels])
            pid = make_pattern_id(category, group_type, group_key, grp_assets)
            front_id, back_id, conf, notes, grp_links = pick_front_back(grp_assets)
            links_out.extend([(pid, aid, role) for _, aid, role in grp_links])

            patterns.append(PatternRec(
                pattern_id=pid,
                category=category,
                pattern_group_type=group_type,
                pattern_group_key=group_key,
                front_asset_id=front_id,
                back_asset_id=back_id,
                confidence=conf,
                notes=notes,
                created_at=now,
                updated_at=now
            ))

    return patterns, links_out


def compute_pattern_counts(patterns: List[PatternRec], links: List[Tuple[str, str, str]], assets: List[Asset]) -> Dict[str, Dict[str, int]]:
    """
    Returns per-pattern counts:
      num_assets, num_images, num_pdfs, num_extra_images, num_other
    """
    asset_by_id = {a.asset_id: a for a in assets}
    counts: Dict[str, Dict[str, int]] = {}

    for pid, aid, role in links:
        c = counts.setdefault(pid, {
            "num_assets": 0,
            "num_images": 0,
            "num_pdfs": 0,
            "num_extra_images": 0,
            "num_other": 0
        })
        c["num_assets"] += 1
        a = asset_by_id.get(aid)
        if not a:
            continue
        if a.file_extension in IMG_EXTS:
            c["num_images"] += 1
            if role == "extra":
                c["num_extra_images"] += 1
        elif a.file_extension in PDF_EXTS:
            c["num_pdfs"] += 1
        else:
            c["num_other"] += 1

    return counts


def write_report_csv(out_csv: Path, patterns: List[PatternRec], assets: List[Asset], links: List[Tuple[str, str, str]]) -> None:
    asset_by_id = {a.asset_id: a for a in assets}
    counts = compute_pattern_counts(patterns, links, assets)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "pattern_id", "category", "group_type", "confidence",
            "front_file", "back_file",
            "has_pdf", "num_assets", "num_images", "num_pdfs", "num_extra_images", "num_other",
            "notes"
        ])

        for p in patterns:
            front = asset_by_id.get(p.front_asset_id).file_name if p.front_asset_id and p.front_asset_id in asset_by_id else ""
            back = asset_by_id.get(p.back_asset_id).file_name if p.back_asset_id and p.back_asset_id in asset_by_id else ""
            c = counts.get(p.pattern_id, {"num_assets": 0, "num_images": 0, "num_pdfs": 0, "num_extra_images": 0, "num_other": 0})
            has_pdf = "1" if c["num_pdfs"] > 0 else "0"

            w.writerow([
                p.pattern_id, p.category, p.pattern_group_type, p.confidence,
                front, back,
                has_pdf, c["num_assets"], c["num_images"], c["num_pdfs"], c["num_extra_images"], c["num_other"],
                p.notes
            ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Path to local mirror of Patterns/data")
    ap.add_argument("--out", default="./out", help="Output folder for DB and report")
    ap.add_argument("--db-name", default="pattern_library.sqlite", help="SQLite DB filename")
    ap.add_argument("--report-name", default="pattern_inventory_report.csv", help="CSV report filename")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not root.exists() or not root.is_dir():
        print(f"ERROR: root folder not found or not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    print(f"[1/4] Scanning assets under: {root}")
    assets = scan_assets(root)
    print(f"  Found {len(assets)} files")

    db_path = out_dir / args.db_name
    print(f"[2/4] Writing SQLite DB: {db_path}")
    conn = sqlite3.connect(str(db_path))
    ensure_tables(conn)
    upsert_assets(conn, assets)

    print(f"[3/4] Grouping into patterns (folder units + file groups)")
    patterns, links = build_patterns(root, assets)
    print(f"  Created {len(patterns)} pattern records")
    upsert_patterns(conn, patterns)
    upsert_links(conn, links)
    conn.close()

    report_path = out_dir / args.report_name
    print(f"[4/4] Writing report CSV: {report_path}")
    write_report_csv(report_path, patterns, assets, links)

    # Quick stats
    missing_front = sum(1 for p in patterns if not p.front_asset_id)
    missing_back = sum(1 for p in patterns if p.front_asset_id and not p.back_asset_id)
    low_conf = sum(1 for p in patterns if p.confidence < 0.7)

    print("\nDone ✅")
    print(f"DB:     {db_path}")
    print(f"Report: {report_path}")
    print(f"Stats:  missing_front={missing_front}, missing_back={missing_back}, low_confidence(<0.7)={low_conf}")
    print("\nTip: Sort the CSV by confidence ascending to review edge cases.")


if __name__ == "__main__":
    main()
