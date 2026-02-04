#!/usr/bin/env python3
"""
L2 Query: Text -> CLIP embedding -> Chroma nearest neighbors

Examples:
  python src/pattern_agent/query_l2.py --q "modern minimal dress, linen" --topk 12 --role front --open_top 3
  python src/pattern_agent/query_l2.py --q "tech office skirt" --topk 12 --role front --prefer_category Skirts --wildcards 2 --open_top 3
"""

import argparse
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import chromadb
import open_clip


def load_clip(model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k"):
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    model.to(device)
    return model, tokenizer, device


@torch.no_grad()
def embed_text(model, tokenizer, device: str, text: str) -> List[float]:
    tokens = tokenizer([text]).to(device)
    feats = model.encode_text(tokens)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats[0].detach().cpu().tolist()


def build_where(role: str) -> Dict:
    """
    Your Chroma version:
      - Allows where={"role":"front"}
      - Requires $and only if you have >=2 conditions
    """
    if role == "any":
        return {}
    return {"role": role}


def safe_open(path: str):
    try:
        subprocess.run(["open", path], check=False)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chroma_dir", default="./out/chroma")
    ap.add_argument("--collection", default="pattern_images")
    ap.add_argument("--q", required=True, help="Text query")
    ap.add_argument("--topk", type=int, default=12)

    ap.add_argument("--role", default="front", choices=["front", "back", "any"])
    ap.add_argument("--prefer_category", default="any", help="Boost this category to the top (e.g., Skirts)")
    ap.add_argument("--wildcards", type=int, default=2, help="How many results can be outside prefer_category")

    ap.add_argument("--open_top", type=int, default=0, help="Open only top N results (0 = none)")
    args = ap.parse_args()

    # ---- CLIP text embedding
    model, tokenizer, device = load_clip()
    q_emb = embed_text(model, tokenizer, device, args.q)

    # ---- Chroma
    client = chromadb.PersistentClient(path=str(Path(args.chroma_dir).resolve()))
    col = client.get_collection(args.collection)

    # Query more than topk so we can dedupe + rerank
    n_results = max(args.topk * 8, 80)

    query_kwargs = {
        "query_embeddings": [q_emb],
        "n_results": n_results,
        "include": ["metadatas", "distances"],  # keep it light
    }

    where = build_where(args.role)
    if where:
        query_kwargs["where"] = where

    res = col.query(**query_kwargs)

    metadatas = res["metadatas"][0]
    distances = res["distances"][0]

    # ---- Deduplicate by pattern_id (keep best match per pattern)
    pool: List[Tuple[dict, float]] = []
    seen = set()
    for md, d in zip(metadatas, distances):
        pid = md.get("pattern_id")
        if not pid or pid in seen:
            continue
        seen.add(pid)
        pool.append((md, d))

    # ---- Prefer category (soft rerank)
    prefer = args.prefer_category
    prefer_any = (prefer == "any")

    preferred: List[Tuple[dict, float]] = []
    others: List[Tuple[dict, float]] = []

    for md, d in pool:
        if (not prefer_any) and (md.get("category") == prefer):
            preferred.append((md, d))
        else:
            others.append((md, d))

    out: List[Tuple[dict, float]] = []

    # Fill from preferred first
    for item in preferred:
        out.append(item)
        if len(out) >= args.topk:
            break

    # Add some wildcards
    if len(out) < args.topk:
        wildcard_budget = min(max(args.wildcards, 0), args.topk - len(out))
        out.extend(others[:wildcard_budget])

    # Fill remaining from others
    if len(out) < args.topk:
        out.extend(others[min(args.wildcards, len(others)) :][: (args.topk - len(out))])

    # ---- Print + open
    print(f"\nQuery: {args.q}\n")
    for i, (md, d) in enumerate(out[: args.topk], start=1):
        pid = md.get("pattern_id", "")
        role = md.get("role", "")
        cat = md.get("category", "")
        file_name = md.get("file_name", "")
        full_path = md.get("full_path", "")

        print(f"{i:02d}. pattern_id={pid} | role={role} | category={cat} | dist={d:.4f}")
        print(f"    file: {file_name}")
        print(f"    path: {full_path}\n")

        if args.open_top > 0 and i <= args.open_top and full_path and os.path.exists(full_path):
            safe_open(full_path)


if __name__ == "__main__":
    main()
