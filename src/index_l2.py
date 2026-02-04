#!/usr/bin/env python3
"""
L2: Build an image+text embedding index using CLIP + Chroma.

- Reads patterns/assets from SQLite (created in L1)
- Embeds front/back images (CLIP image embeddings)
- Stores in Chroma persistent DB
- Also stores metadata to map results back to pattern_id + file paths

Run:
  python src/pattern_agent/index_l2.py \
    --db "./out/pattern_library.sqlite" \
    --chroma_dir "./out/chroma" \
    --collection "pattern_images" \
    --embed_roles front back
"""

import argparse
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict

import torch
from PIL import Image
from tqdm import tqdm
import chromadb
import open_clip


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class ImgRow:
    pattern_id: str
    category: str
    role: str          # front/back
    asset_id: str
    full_path: str
    file_name: str


def fetch_image_rows(sqlite_path: Path, roles: List[str]) -> List[ImgRow]:
    roles = [r.lower() for r in roles]
    conn = sqlite3.connect(str(sqlite_path))
    cur = conn.cursor()

    # We trust the link roles from L1: 'front' and 'back'
    q = f"""
    SELECT
        p.pattern_id,
        p.category,
        l.role,
        a.asset_id,
        a.full_path,
        a.file_name
    FROM patterns p
    JOIN pattern_asset_links l ON p.pattern_id = l.pattern_id
    JOIN pattern_assets a ON a.asset_id = l.asset_id
    WHERE l.role IN ({",".join(["?"]*len(roles))})
    """
    cur.execute(q, roles)
    rows = cur.fetchall()
    conn.close()

    out: List[ImgRow] = []
    for pattern_id, category, role, asset_id, full_path, file_name in rows:
        ext = Path(full_path).suffix.lower()
        if ext in IMG_EXTS and os.path.exists(full_path):
            out.append(ImgRow(
                pattern_id=pattern_id,
                category=category,
                role=role,
                asset_id=asset_id,
                full_path=full_path,
                file_name=file_name
            ))
    return out


def load_clip(model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k"):
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    model.to(device)
    return model, preprocess, tokenizer, device


@torch.no_grad()
def embed_images(model, preprocess, device: str, paths: List[str], batch_size: int) -> torch.Tensor:
    embs = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i+batch_size]
        imgs = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                imgs.append(preprocess(img))
            except Exception:
                imgs.append(None)

        valid_idx = [j for j, x in enumerate(imgs) if x is not None]
        if not valid_idx:
            continue

        x = torch.stack([imgs[j] for j in valid_idx]).to(device)
        feats = model.encode_image(x)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        # pad back to original batch order with NaNs, so we can align
        out = torch.full((len(batch_paths), feats.shape[-1]), float("nan"), device="cpu")
        out_valid = feats.detach().to("cpu")
        for k, j in enumerate(valid_idx):
            out[j] = out_valid[k]
        embs.append(out)

    return torch.cat(embs, dim=0) if embs else torch.empty((0, 512))


def chunked(iterable: List[ImgRow], n: int) -> Iterable[List[ImgRow]]:
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Path to pattern_library.sqlite")
    ap.add_argument("--chroma_dir", default="./out/chroma", help="Chroma persistence directory")
    ap.add_argument("--collection", default="pattern_images", help="Chroma collection name")
    ap.add_argument("--embed_roles", nargs="+", default=["front"], help="Roles to embed: front back")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--upsert_batch", type=int, default=256, help="How many items per Chroma upsert")
    args = ap.parse_args()

    sqlite_path = Path(args.db).resolve()
    if not sqlite_path.exists():
        raise FileNotFoundError(f"SQLite DB not found: {sqlite_path}")

    rows = fetch_image_rows(sqlite_path, args.embed_roles)
    if not rows:
        print("No image rows found to embed. Check roles and file paths.")
        return

    print(f"Found {len(rows)} images to embed (roles={args.embed_roles}).")

    model, preprocess, tokenizer, device = load_clip()
    print(f"Using device: {device}")

    chroma_client = chromadb.PersistentClient(path=str(Path(args.chroma_dir).resolve()))
    col = chroma_client.get_or_create_collection(name=args.collection)

    # To support re-runs, we use a deterministic vector id:
    # pattern_id|role|asset_id
    def make_id(r: ImgRow) -> str:
        return f"{r.pattern_id}|{r.role}|{r.asset_id}"

    total_upserted = 0
    for batch in tqdm(list(chunked(rows, args.upsert_batch)), desc="Embedding+Upserting"):
        ids = [make_id(r) for r in batch]
        paths = [r.full_path for r in batch]

        emb = embed_images(model, preprocess, device, paths, batch_size=args.batch_size)  # shape (N, D)
        if emb.shape[0] != len(batch):
            # should not happen; safe guard
            emb = emb[:len(batch)]

        # Filter invalid rows (NaN embeddings from failed image loads)
        valid = []
        for i in range(len(batch)):
            v = emb[i]
            if torch.isnan(v).any():
                continue
            valid.append(i)

        if not valid:
            continue

        valid_ids = [ids[i] for i in valid]
        valid_embs = [emb[i].tolist() for i in valid]
        metadatas = [{
            "pattern_id": batch[i].pattern_id,
            "category": batch[i].category,
            "role": batch[i].role,
            "asset_id": batch[i].asset_id,
            "file_name": batch[i].file_name,
            "full_path": batch[i].full_path,
        } for i in valid]

        # A short doc string is fine (optional). We’ll store file names.
        documents = [batch[i].file_name for i in valid]

        col.upsert(ids=valid_ids, embeddings=valid_embs, metadatas=metadatas, documents=documents)
        total_upserted += len(valid)

    print(f"Done ✅ Upserted {total_upserted} embeddings into Chroma collection '{args.collection}'.")
    print(f"Chroma dir: {Path(args.chroma_dir).resolve()}")


if __name__ == "__main__":
    main()
