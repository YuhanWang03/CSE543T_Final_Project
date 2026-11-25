"""Frozen retrievers: BM25 (rank-bm25) and E5-small-v2 dense; full-corpus score matrices."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm


def bm25_score_matrix(doc_tokens, query_tokens, k1: float, b: float, epsilon: float) -> np.ndarray:
    """[n_queries, n_docs] float32 BM25Okapi scores over the whole corpus."""
    bm25 = BM25Okapi(doc_tokens, k1=k1, b=b, epsilon=epsilon)
    out = np.zeros((len(query_tokens), len(doc_tokens)), dtype=np.float32)
    for i, q in enumerate(tqdm(query_tokens, desc="bm25", mininterval=2)):
        if q:
            out[i] = bm25.get_scores(q)
    return out


def resolve_model(name: str, revision: str | None) -> tuple[Path, str]:
    """Download (or reuse) the model snapshot; return local path and resolved commit hash."""
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(name, revision=revision,
                                  allow_patterns=["*.json", "*.txt", "model.safetensors"]))
    if not (path / "model.safetensors").exists():
        path = Path(snapshot_download(name, revision=revision,
                                      allow_patterns=["*.json", "*.txt", "pytorch_model.bin"]))
    return path, path.name


def load_tokenizer(path: Path):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    assert tok.is_fast, "a fast tokenizer is required for offset mapping"
    return tok


def encode(texts, tokenizer, model_path: Path, max_len: int, batch_size: int, device: str) -> np.ndarray:
    """Mean-pooled, L2-normalised E5 embeddings (float32)."""
    import torch
    from transformers import AutoModel

    if device == "cuda" and not torch.cuda.is_available():
        print("[dense] CUDA not available, using CPU")
        device = "cpu"
    model = AutoModel.from_pretrained(model_path).to(device).eval()
    order = np.argsort([len(t) for t in texts])  # length-sorted batches: faster, result unchanged
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    with torch.inference_mode():
        for s in tqdm(range(0, len(texts), batch_size), desc="encode", mininterval=2):
            idx = order[s: s + batch_size]
            batch = tokenizer([texts[i] for i in idx], max_length=max_len, truncation=True,
                              padding=True, return_tensors="pt").to(device)
            h = model(**batch).last_hidden_state
            m = batch["attention_mask"].unsqueeze(-1).to(h.dtype)
            emb = (h * m).sum(1) / m.sum(1)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            out[idx] = emb.float().cpu().numpy()
    return out
