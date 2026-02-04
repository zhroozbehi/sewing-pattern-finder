# Sewing Pattern Finder 🧵

Search your sewing pattern library by *vibe*, not just keywords.

> "modern minimal linen dress" → finds matching patterns from your collection

## The Problem

Sewing pattern collections grow fast. PDFs, photos of vintage patterns, envelope scans — scattered across folders. Finding "that one 70s wrap dress" means clicking through hundreds of files.

## The Solution

This tool indexes your pattern images and lets you search using natural language descriptions. It uses CLIP embeddings to understand what patterns *look like*, not just their filenames.

**Example searches:**
- "boxy oversized jacket"
- "romantic blouse puff sleeves"
- "minimalist A-line skirt"

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│  1. INDEX                                                   │
│  Scan folders → Pair front/back images → SQLite catalog     │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  2. EMBED                                                   │
│  CLIP model → Generate image embeddings → ChromaDB          │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  3. SEARCH                                                  │
│  Text query → CLIP text embedding → Vector similarity       │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  4. UI                                                      │
│  Streamlit app → Browse results → Open/reveal files         │
└─────────────────────────────────────────────────────────────┘
```

## Features

- **Natural language search** — describe what you want, not filenames
- **Smart pairing** — automatically groups front/back pattern images
- **Category filtering** — prefer results from specific categories (Dress, Skirts, etc.)
- **Quick preview** — open images directly from search results

## Tech Stack

- **Python**
- **CLIP (OpenCLIP)** — image + text embeddings
- **ChromaDB** — vector database
- **SQLite** — pattern metadata catalog
- **Streamlit** — web UI

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Index your patterns

```bash
python src/index_l1.py --root "/path/to/your/patterns" --out "./out"
```

### 3. Generate embeddings

```bash
python src/index_l2.py --db "./out/pattern_library.sqlite" --chroma_dir "./out/chroma"
```

### 4. Run the app

```bash
streamlit run src/app.py
```

## Folder Structure

```
sewing-pattern-finder/
├── src/
│   ├── index_l1.py      # Scan & catalog patterns
│   ├── index_l2.py      # Generate CLIP embeddings
│   ├── query_l2.py      # CLI search
│   └── app.py           # Streamlit UI
├── out/                  # Generated DB & embeddings (gitignored)
├── requirements.txt
└── README.md
```

## Screenshots

*Coming soon*

## Future Ideas

- [ ] LLM-generated pattern descriptions
- [ ] "Similar to this" image-to-image search
- [ ] Favorites & saved searches
- [ ] Cloud deployment

---

Built by [Zahra](https://github.com/zhroozbehi)
