"""
Embed products.jsonl and index into Qdrant.

Creates a Qdrant collection with both dense and sparse vector fields.
On --rebuild, drops the existing collection and re-creates it from scratch.
Use --rebuild whenever you change embedding models or the collection schema.

Usage:
  # First time or after catalog changes (upsert, safe to run repeatedly)
  python scripts/embed_catalog.py

  # After changing DENSE_MODEL or SPARSE_MODEL (drops + rebuilds collection)
  python scripts/embed_catalog.py --rebuild

  # With non-default models
  DENSE_MODEL=BAAI/bge-large-en-v1.5 python scripts/embed_catalog.py --rebuild

  # Point at a different catalog file
  python scripts/embed_catalog.py --catalog data/catalog/products.jsonl --rebuild
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from pipeline.embeddings import FastEmbedProvider, default_provider
from pipeline.models import Product

COLLECTION_NAME = "products"
BATCH_SIZE = 512  # Products per embedding + upsert batch. GPU can handle large batches.


def get_qdrant_client():
    from qdrant_client import QdrantClient
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key = os.getenv("QDRANT_API_KEY") or None  # empty string → None for local dev
    return QdrantClient(url=url, api_key=api_key, timeout=60)


def create_collection(client, dense_dimensions: int) -> None:
    """
    Create the Qdrant collection with dense + sparse vector fields.

    The schema is fixed at creation time. To change dense_dimensions,
    you must --rebuild (drop + recreate).
    """
    from qdrant_client.models import (
        Distance,
        SparseVectorParams,
        VectorParams,
    )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(
                size=dense_dimensions,
                distance=Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams()
            # Qdrant handles sparse vector storage and dot-product search natively.
            # No dimension declaration needed for sparse vectors.
        },
    )
    print(f"  Created collection '{COLLECTION_NAME}' (dense_dims={dense_dimensions})")


def collection_exists(client) -> bool:
    existing = [c.name for c in client.get_collections().collections]
    return COLLECTION_NAME in existing


def drop_collection(client) -> None:
    client.delete_collection(COLLECTION_NAME)
    print(f"  Dropped collection '{COLLECTION_NAME}'")


def load_products(catalog_path: Path) -> list[Product]:
    products = []
    with catalog_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                products.append(Product.model_validate_json(line))
            except Exception as e:
                print(f"  [warn] Skipping malformed record: {e}")
    return products


def validate_search_texts(products: list[Product]) -> None:
    """Fail fast if any product is missing pre-computed search texts."""
    missing = [p.id for p in products if not p.dense_text or not p.sparse_text]
    if missing:
        print(
            f"\n[error] {len(missing)} products are missing search texts. "
            f"Run ingest_catalog.py first.\nAffected IDs: {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
        sys.exit(1)


def upsert_batch(
    client,
    products: list[Product],
    embeddings,
    start_id: int,
) -> None:
    from qdrant_client.models import PointStruct, SparseVector

    points = []
    for i, (product, emb) in enumerate(zip(products, embeddings)):
        points.append(
            PointStruct(
                id=start_id + i,
                vector={
                    "dense": emb.dense.values,
                    "sparse": SparseVector(
                        indices=emb.sparse.indices,
                        values=emb.sparse.values,
                    ),
                },
                payload=product.model_dump(
                    exclude={"dense_text", "sparse_text"}
                    # Exclude search texts from payload — they're large and not needed at query time.
                    # The payload is what the retriever returns as product metadata.
                ),
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed catalog and index into Qdrant")
    parser.add_argument(
        "--catalog",
        default="data/catalog/products.jsonl",
        help="Path to normalized products.jsonl",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Drop and recreate the Qdrant collection before indexing. "
             "Required when changing embedding models or collection schema.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Products per embedding batch (default: {BATCH_SIZE})",
    )
    args = parser.parse_args()

    catalog_path = Path(args.catalog)
    if not catalog_path.exists():
        print(f"[error] Catalog not found: {catalog_path}")
        print("Run: python scripts/ingest_catalog.py")
        sys.exit(1)

    print(f"Loading catalog: {catalog_path}")
    products = load_products(catalog_path)
    print(f"  {len(products)} products")
    validate_search_texts(products)

    print("\nInitializing embedding provider...")
    provider = default_provider()
    print(f"  Dense:  {os.getenv('DENSE_MODEL', 'BAAI/bge-small-en-v1.5')} ({provider.dense_dimensions} dims)")
    print(f"  Sparse: {os.getenv('SPARSE_MODEL', 'prithivida/Splade_PP_en_v1')}")
    print("  (Models download on first use — this may take a moment)\n")

    client = get_qdrant_client()

    if args.rebuild:
        if collection_exists(client):
            drop_collection(client)
        create_collection(client, provider.dense_dimensions)
    elif not collection_exists(client):
        create_collection(client, provider.dense_dimensions)
    else:
        print(f"Collection '{COLLECTION_NAME}' exists — upserting (use --rebuild to reset)")

    # --- Embed and index in batches ---
    total = len(products)
    indexed = 0

    for batch_start in range(0, total, args.batch_size):
        batch = products[batch_start : batch_start + args.batch_size]
        dense_texts = [p.dense_text for p in batch]
        sparse_texts = [p.sparse_text for p in batch]

        embeddings = provider.embed_batch(dense_texts, sparse_texts)
        upsert_batch(client, batch, embeddings, start_id=batch_start)

        indexed += len(batch)
        pct = indexed / total * 100
        print(f"  [{pct:5.1f}%] {indexed}/{total} products indexed", flush=True)

    print(f"\n\nDone. {indexed} products indexed into '{COLLECTION_NAME}'.")
    print(f"Qdrant: {os.getenv('QDRANT_URL', 'http://localhost:6333')}")


if __name__ == "__main__":
    main()
