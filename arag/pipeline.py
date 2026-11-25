"""Stages that build cached artefacts: prepare -> retrieve -> features."""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from importlib import metadata

import numpy as np

from .config import read_json, record_decision, write_json
from .data import download_and_extract, duplicate_query_texts, load_dataset, md5sum, verify
from .features import document_frequencies, jaccard_distance, query_features, rbo_distance
from .fusion import rank, standardize
from .retrievers import bm25_score_matrix, encode, load_tokenizer, resolve_model
from .text import bm25_tokenize, build_visible_texts

TOP_N = 100  # cached top-N per retriever (hard negatives use ranks 21-100)


def get_dataset(cfg):
    return load_dataset(cfg["paths"]["data_dir"] / "nfcorpus")


# ---------------------------------------------------------------- prepare
def stage_prepare(cfg):
    folder = download_and_extract(cfg)
    ds = load_dataset(folder)
    stats = verify(ds, cfg)
    dup = duplicate_query_texts(ds)
    res = cfg["paths"]["results_dir"]
    write_json(res / "data_check.json", {"verification": stats, "duplicate_query_texts": dup["shared_texts"],
                                         "test_query_ids_overlapping_train": dup["test_query_ids_overlapping_train"]})
    record_decision(cfg, "dedup_test_rows", dup["dedup_test_rows"])
    print(f"[prepare] verified: {stats['corpus']} docs, {stats['queries']} queries; "
          f"splits " + ", ".join(f"{s}={v['queries']}" for s, v in stats["splits"].items()))
    print(f"[prepare] shared normalised query texts: {dup['shared_texts']}")

    model_path, revision = resolve_model(cfg["dense"]["model"], cfg["dense"]["revision"])
    tok = load_tokenizer(model_path)
    visible, truncated, tstats = build_visible_texts(ds.titles, ds.texts, tok, cfg["text"]["max_tokens"],
                                                     cfg["dense"]["passage_prefix"])
    tstats["model_revision"] = revision
    cache = cfg["paths"]["cache_dir"]
    with open(cache / "visible_texts.jsonl", "w", encoding="utf-8") as f:
        for did, t, tr in zip(ds.doc_ids, visible, truncated):
            f.write(json.dumps({"doc_id": did, "truncated": bool(tr), "text": t}, ensure_ascii=False) + "\n")
    write_json(res / "text_stats.json", tstats)
    print(f"[prepare] visible text: {tstats['n_truncated']} of {ds.n_docs} docs truncated "
          f"({tstats['frac_truncated']:.1%}); model revision {revision}")
    return {"model_revision": revision, "data_md5": md5sum(cfg["paths"]["data_dir"] / "nfcorpus.zip")}


def load_visible(cfg):
    with open(cfg["paths"]["cache_dir"] / "visible_texts.jsonl", encoding="utf-8") as f:
        return [json.loads(l)["text"] for l in f]


# ---------------------------------------------------------------- retrieve
def stage_retrieve(cfg):
    ds = get_dataset(cfg)
    visible = load_visible(cfg)
    cache = cfg["paths"]["cache_dir"]
    b = cfg["bm25"]

    t0 = time.perf_counter()
    doc_tokens = [bm25_tokenize(t) for t in visible]
    q_tokens = [bm25_tokenize(q) for q in ds.query_texts]
    B = bm25_score_matrix(doc_tokens, q_tokens, b["k1"], b["b"], b["epsilon"])
    t_bm25 = time.perf_counter() - t0

    dcfg = cfg["dense"]
    model_path, revision = resolve_model(dcfg["model"], dcfg["revision"])
    tok = load_tokenizer(model_path)
    t0 = time.perf_counter()
    demb = encode([dcfg["passage_prefix"] + t for t in visible], tok, model_path, cfg["text"]["max_tokens"],
                  dcfg["batch_size"], dcfg["device"])
    t_doc = time.perf_counter() - t0
    t0 = time.perf_counter()
    qemb = encode([dcfg["query_prefix"] + q for q in ds.query_texts], tok, model_path, cfg["text"]["max_tokens"],
                  dcfg["batch_size"], dcfg["device"])
    D = (qemb @ demb.T).astype(np.float32)
    t_q = time.perf_counter() - t0

    np.save(cache / "scores_bm25.npy", B)
    np.save(cache / "scores_dense.npy", D)
    np.save(cache / "emb_docs.npy", demb)
    write_json(cache / "rows.json", {"doc_ids": ds.doc_ids, "query_ids": ds.query_ids})
    costs = {"bm25_index_and_score_all_queries_s": t_bm25, "dense_doc_encoding_s": t_doc,
             "dense_query_encoding_and_scoring_s": t_q, "n_queries": ds.n_queries, "n_docs": ds.n_docs}
    write_json(cfg["paths"]["results_dir"] / "retrieval_costs.json", costs)
    print(f"[retrieve] BM25 {t_bm25:.1f}s; dense docs {t_doc:.1f}s; dense queries {t_q:.1f}s; "
          f"empty BM25 queries: {sum(1 for q in q_tokens if not q)}")
    return {"model_revision": revision}


def load_scores(cfg):
    cache = cfg["paths"]["cache_dir"]
    return np.load(cache / "scores_bm25.npy"), np.load(cache / "scores_dense.npy")


# ---------------------------------------------------------------- features
def stage_features(cfg):
    ds = get_dataset(cfg)
    visible = load_visible(cfg)
    B, D = load_scores(cfg)
    eps = cfg["fusion"]["std_eps"]
    Bz, deg_b = standardize(B, eps)
    Dz, deg_d = standardize(D, eps)
    top_b, top_d = rank(B, TOP_N), rank(D, TOP_N)
    dv = cfg["divergence"]
    g_j = jaccard_distance(top_b[:, : dv["jaccard_k"]], top_d[:, : dv["jaccard_k"]])
    g_r = rbo_distance(top_b[:, : dv["rbo_depth"]], top_d[:, : dv["rbo_depth"]], dv["rbo_p"])
    doc_tokens = [bm25_tokenize(t) for t in visible]
    q_tokens = [bm25_tokenize(q) for q in ds.query_texts]
    feats = query_features(q_tokens, document_frequencies(doc_tokens), ds.n_docs, B, Bz, Dz, top_b, top_d, g_j, g_r)
    cache = cfg["paths"]["cache_dir"]
    np.savez(cache / "features.npz", **feats)
    np.save(cache / "top_bm25.npy", top_b.astype(np.int32))
    np.save(cache / "top_dense.npy", top_d.astype(np.int32))
    np.savez(cache / "degenerate.npz", bm25=deg_b, dense=deg_d)
    print(f"[features] degenerate rows: bm25={int(deg_b.sum())}, dense={int(deg_d.sum())}; "
          f"features: {', '.join(feats)}")
    return {}


# ---------------------------------------------------------------- run info
def write_run_info(cfg, stage_info: dict, stages: list[str]):
    # Versions of the modules actually imported (the base conda env carries stale dist-info metadata).
    import importlib
    pkgs = {}
    for mod in ["numpy", "scipy", "torch", "torchvision", "transformers", "tokenizers", "huggingface_hub",
                "matplotlib", "yaml"]:
        try:
            pkgs[mod] = getattr(importlib.import_module(mod), "__version__", None)
        except Exception:
            pkgs[mod] = None
    for dist in ["rank-bm25", "pytrec-eval-terrier"]:
        try:
            pkgs[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            pkgs[dist] = None
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        gpu = None
    path = cfg["paths"]["results_dir"] / "run_info.json"
    log = read_json(path) if path.exists() else {"runs": []}
    log["runs"].append({
        "time": time.strftime("%Y-%m-%d %H:%M:%S"), "stages": stages, "python": sys.version.split()[0],
        "platform": platform.platform(), "gpu": gpu, "threads": os.environ.get("OMP_NUM_THREADS"),
        "seed": cfg["runtime"]["seed"], "packages": pkgs, "git_commit": None, **stage_info,
    })
    write_json(path, log)
