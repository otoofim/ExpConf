# CVPR Explorer

Interactive 2D browser for CVPR, ICCV, ECCV, and WACV papers. Papers with similar abstracts cluster together on a semantic map powered by sentence embeddings and UMAP/t-SNE.

## Features

- 2D semantic scatter plot — related papers cluster automatically
- Click any point to read the title, abstract, and open the PDF
- Filter by conference/year, research field, or keyword search
- Colour-coded by research area (diffusion, ViT, NeRF, detection, etc.)
- Embeddings via any OpenAI-compatible API (`text-embedding-3-large`, `qwen3-embedding-8b`, etc.)

## Quick Start

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure API credentials

```bash
cp .env.example .env
# Edit .env and set API_BASE_URL and API_KEY
```

### 3. Scrape papers

```bash
# CVPR 2024 (≈2500 papers, takes ~30 min with default 0.3 s delay)
uv run python download.py --conf CVPR --year 2024

# Faster test run — first 100 papers only
uv run python download.py --conf CVPR --year 2024 --max_papers 100
```

Supported conferences and years:
| Conference | Years |
|---|---|
| CVPR | 2013–2024 |
| ICCV | 2013, 2015, 2017, 2019, 2021, 2023 |
| ECCV | 2018, 2020, 2022 |
| WACV | 2020–2025 |

### 4. Generate embeddings

```bash
# Default: text-embedding-3-large
uv run python embed.py --conf CVPR --year 2024

# Use a different model from your API
uv run python embed.py --conf CVPR --year 2024 --model qwen3-embedding-8b
```

### 5. Run the app

```bash
uv run python app.py
# Open http://localhost:8050
```

Or with gunicorn for production:
```bash
uv run gunicorn app:server -b 0.0.0.0:8050 --workers 2
```

### Docker

```bash
docker compose up --build
# Open http://localhost:8050
```

Place your `data/` files on the host — they are volume-mounted into the container.

## Data Pipeline

```
download.py  →  data/cvpr_2024_papers.json        (titles, abstracts, PDF links)
embed.py     →  data/cvpr_2024_embeddings.npy      (float32 embedding matrix via API)
app.py       →  UMAP/t-SNE 2D coords at startup, then serves Dash app
```

## Embedding models

Any model from your API can be used via `--model`. Recommended choices:

| Model | Dims | Notes |
|---|---|---|
| `text-embedding-3-large` | 3072 | Default; high quality |
| `qwen3-embedding-8b` | 4096 | Strong open-weights alternative |
| `bge-m3` | 1024 | Multilingual, fast |
| `text-embedding-ada-002` | 1536 | Legacy fallback |

## Project Structure

```
cvpr-explorer/
├── download.py          # Stage 1: scrape CVF Open Access
├── embed.py             # Stage 2: generate sentence embeddings
├── app.py               # Stage 3: 2D visualisation + Dash web app
├── pyproject.toml       # Project metadata and dependencies
├── uv.lock              # Locked dependency graph
├── .python-version      # Pinned Python version (3.12)
├── Dockerfile
├── docker-compose.yml
└── data/                # JSON + .npy files land here (git-ignored)
```

## Acknowledgements

Inspired by [dataplayer12/cvpr-explorer](https://github.com/dataplayer12/cvpr-explorer).
Paper data sourced from [CVF Open Access](https://openaccess.thecvf.com/).
