import json
import numpy as np
from pathlib import Path

root = Path("data/test_embeddings")

checks = [
    ("small_fixed", root / "small_fixed" / "metadata.jsonl", root / "small_fixed" / "embeddings.npy"),
    ("medium_overlap", root / "medium_overlap" / "metadata.jsonl", root / "medium_overlap" / "embeddings.npy"),
    ("hier_parent", root / "hierarchical" / "parent_metadata.jsonl", root / "hierarchical" / "parent_embeddings.npy"),
    ("hier_child", root / "hierarchical" / "child_metadata.jsonl", root / "hierarchical" / "child_embeddings.npy"),
]

for name, meta_path, emb_path in checks:
    rows = [json.loads(line) for line in meta_path.open("r", encoding="utf-8") if line.strip()]
    embs = np.load(emb_path)

    print(f"\n{name}")
    print("metadata rows:", len(rows))
    print("embedding shape:", embs.shape)
    print("first chunk_id:", rows[0]["chunk_id"])
    print("first embedding_row:", rows[0]["embedding_row"])
    print("last embedding_row:", rows[-1]["embedding_row"])