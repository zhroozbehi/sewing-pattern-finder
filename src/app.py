import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import chromadb
import open_clip
import streamlit as st
import torch
from PIL import Image


# ---------- Config ----------
DEFAULT_CHROMA_DIR = "./out/chroma"
DEFAULT_COLLECTION = "pattern_images"
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
INTERNAL_N = 120  # query a bigger pool so dedupe + rerank has room


# ---------- Helpers ----------
def safe_open(path: str):
    try:
        subprocess.run(["open", path], check=False)
    except Exception:
        pass


def safe_reveal(path: str):
    try:
        subprocess.run(["open", "-R", path], check=False)
    except Exception:
        pass


def load_thumb(path: str, max_px: int = 1200):
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_px, max_px))
    return img


def build_where(role: str) -> Dict:
    if role == "any":
        return {}
    return {"role": role}


def dedupe_by_pattern_id(metadatas: List[dict], distances: List[float]) -> List[Tuple[dict, float]]:
    seen = set()
    out = []
    for md, d in zip(metadatas, distances):
        pid = md.get("pattern_id")
        if not pid or pid in seen:
            continue
        seen.add(pid)
        out.append((md, d))
    return out


def rerank_prefer_category(pool: List[Tuple[dict, float]], prefer_category: str, wildcards: int, topk: int):
    if prefer_category == "any":
        # just take topk
        return pool[:topk]

    preferred = []
    others = []
    for md, d in pool:
        if md.get("category") == prefer_category:
            preferred.append((md, d))
        else:
            others.append((md, d))

    out = []
    # preferred first
    for item in preferred:
        out.append(item)
        if len(out) >= topk:
            return out

    # some wildcards
    wildcard_budget = min(max(wildcards, 0), topk - len(out))
    out.extend(others[:wildcard_budget])

    # fill remaining
    if len(out) < topk:
        out.extend(others[wildcard_budget:][: (topk - len(out))])

    return out


# ---------- Cached resources ----------
@st.cache_resource
def get_collection(chroma_dir: str, collection_name: str):
    client = chromadb.PersistentClient(path=str(Path(chroma_dir).resolve()))
    return client.get_collection(collection_name)


@st.cache_resource
def load_clip():
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model.eval()
    model.to(device)
    return model, tokenizer, device


@torch.no_grad()
def embed_text(model, tokenizer, device: str, text: str) -> List[float]:
    tokens = tokenizer([text]).to(device)
    feats = model.encode_text(tokens)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats[0].detach().cpu().tolist()


# ---------- UI ----------
st.set_page_config(page_title="Pattern Agent", layout="wide")
st.title("🧵 Pattern Agent")
st.caption("Search your pattern library by vibe (CLIP + Chroma).")

with st.sidebar:
    st.header("Index")
    chroma_dir = st.text_input("Chroma directory", value=DEFAULT_CHROMA_DIR)
    collection_name = st.text_input("Collection", value=DEFAULT_COLLECTION)

    st.header("Search options")
    role = st.selectbox("Use images from", options=["front", "back", "any"], index=0)

    prefer_category = st.text_input("Prefer category (optional)", value="any",
                                   help="Exact match like: Skirts, Dress, Jackets, Mix ... or 'any'")
    wildcards = st.slider("Wildcards (outside preferred category)", 0, 6, 2, 1)

    topk = st.slider("Results to show", 6, 48, 12, 6)
    grid_cols = st.selectbox("Grid columns", [2, 3, 4, 6], index=2)

    st.header("Auto-open")
    open_top = st.slider("Open top N", 0, min(12, topk), 0, 1,
                         help="Opens top N images in Preview/Finder on macOS")


q = st.text_input(
    "Describe what you want to sew",
    value="modern minimal, tech office",
    help='Examples: "linen midi skirt minimalist", "boxy jacket oversized", "romantic blouse puff sleeves"'
)

search = st.button("Search", type="primary")

# Auto run once (nice UX)
if "ran_once" not in st.session_state:
    st.session_state["ran_once"] = True
    search = True

if search:
    # Load model + collection
    model, tokenizer, device = load_clip()
    col = get_collection(chroma_dir, collection_name)

    # Embed and query
    q_emb = embed_text(model, tokenizer, device, q)

    query_kwargs = {
        "query_embeddings": [q_emb],
        "n_results": INTERNAL_N,
        "include": ["metadatas", "distances"],
    }
    where = build_where(role)
    if where:
        query_kwargs["where"] = where

    res = col.query(**query_kwargs)

    metadatas = res["metadatas"][0]
    distances = res["distances"][0]

    pool = dedupe_by_pattern_id(metadatas, distances)
    results = rerank_prefer_category(pool, prefer_category, wildcards, topk)

    if not results:
        st.warning("No results. Try role='any' or a different query.")
        st.stop()

    # Auto-open top N
    if open_top > 0:
        for i, (md, _d) in enumerate(results[:open_top], start=1):
            p = md.get("full_path", "")
            if p and os.path.exists(p):
                safe_open(p)

    st.subheader("Results")
    cols = st.columns(grid_cols, gap="large")

    for idx, (md, d) in enumerate(results, start=1):
        pid = md.get("pattern_id", "")
        cat = md.get("category", "")
        rrole = md.get("role", "")
        file_name = md.get("file_name", "")
        full_path = md.get("full_path", "")

        col_i = cols[(idx - 1) % grid_cols]
        with col_i:
            # image
            if full_path and os.path.exists(full_path):
                try:
                    st.image(load_thumb(full_path), use_container_width=True)
                except Exception:
                    st.write("⚠️ Could not load image")
            else:
                st.write("⚠️ Missing file")

            st.markdown(f"**{idx:02d}. {pid}**")
            st.caption(f"{cat} • {rrole} • dist={d:.4f}")
            st.code(file_name, language=None)

            b1, b2 = st.columns(2)
            with b1:
                if st.button("Open", key=f"open_{pid}_{idx}"):
                    if full_path:
                        safe_open(full_path)
            with b2:
                if st.button("Reveal", key=f"reveal_{pid}_{idx}"):
                    if full_path:
                        safe_reveal(full_path)

    st.divider()
    st.write("Next upgrades: ⭐ favorites, saved searches, and a 'modern styling ideas' panel per result.")
