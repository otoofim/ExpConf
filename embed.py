"""
Generate embeddings for scraped papers via an OpenAI-compatible API.

Config (via .env or environment variables):
    API_BASE_URL   e.g. https://your-api/v1
    API_KEY        your API key

Usage:
    python embed.py --conf CVPR --year 2024
    python embed.py --conf CVPR --year 2024 --model qwen3-embedding-8b

Input:  data/{conf}_{year}_papers.json
Output: data/{conf}_{year}_embeddings.npy
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

DEFAULT_MODEL = "text-embedding-3-large"
# Max texts per API call — stay safely under the 8k-token-per-input limit
BATCH_SIZE = 32


def make_client() -> OpenAI:
    base_url = os.environ.get("API_BASE_URL")
    api_key = os.environ.get("API_KEY", "placeholder")
    if not base_url:
        raise ValueError("API_BASE_URL not set. Add it to .env or export it.")
    return OpenAI(base_url=base_url, api_key=api_key)


def embed_batch(client: OpenAI, texts: list[str], model: str) -> list[list[float]]:
    response = client.embeddings.create(input=texts, model=model)
    # Sort by index in case the API returns out-of-order
    items = sorted(response.data, key=lambda e: e.index)
    return [item.embedding for item in items]


def embed_papers(
    papers: list[dict],
    model: str,
    batch_size: int = BATCH_SIZE,
    retry_delay: float = 2.0,
) -> np.ndarray:
    client = make_client()
    texts = [p.get("abstract") or p.get("title") or "" for p in papers]

    all_embeddings: list[list[float]] = []
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

    for batch in tqdm(batches, desc="Embedding"):
        for attempt in range(3):
            try:
                all_embeddings.extend(embed_batch(client, batch, model))
                break
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"  Retry {attempt + 1}/3 after error: {e}")
                time.sleep(retry_delay * (attempt + 1))

    return np.array(all_embeddings, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Embed paper abstracts via API")
    parser.add_argument("--conf", default="CVPR")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Embedding model id (e.g. text-embedding-3-large, qwen3-embedding-8b)")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    data_dir = Path("data")
    papers_path = data_dir / f"{args.conf.lower()}_{args.year}_papers.json"
    out_path = data_dir / f"{args.conf.lower()}_{args.year}_embeddings.npy"

    if not papers_path.exists():
        print(f"Papers file not found: {papers_path}")
        print("Run download.py first.")
        return

    if out_path.exists():
        print(f"Embeddings already exist: {out_path}. Delete to re-embed.")
        return

    with open(papers_path) as f:
        papers = json.load(f)
    print(f"Loaded {len(papers)} papers from {papers_path}")

    print(f"Embedding with model '{args.model}' (batch_size={args.batch_size})...")
    embeddings = embed_papers(papers, args.model, args.batch_size)

    np.save(out_path, embeddings)
    print(f"Saved embeddings {embeddings.shape} → {out_path}")


if __name__ == "__main__":
    main()
