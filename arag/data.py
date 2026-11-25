"""Download, verify and load BEIR NFCorpus (plan v3 §3)."""
from __future__ import annotations

import csv
import hashlib
import json
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

SPLITS = ("train", "dev", "test")


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_and_extract(cfg: dict) -> Path:
    """Download nfcorpus.zip from the official BEIR URL, check MD5, extract safely."""
    data_dir: Path = cfg["paths"]["data_dir"]
    zpath = data_dir / "nfcorpus.zip"
    if not zpath.exists():
        tmp = zpath.with_suffix(".part")
        print(f"[data] downloading {cfg['data']['url']}")
        urllib.request.urlretrieve(cfg["data"]["url"], tmp)
        tmp.replace(zpath)
    digest = md5sum(zpath)
    if digest != cfg["data"]["md5"]:
        raise RuntimeError(f"MD5 mismatch for {zpath}: {digest} != {cfg['data']['md5']}")
    out = data_dir / "nfcorpus"
    if not (out / "corpus.jsonl").exists():
        root = data_dir.resolve()
        with zipfile.ZipFile(zpath) as z:
            for name in z.namelist():  # refuse path traversal
                target = (data_dir / name).resolve()
                if root not in target.parents and target != root:
                    raise RuntimeError(f"unsafe path in zip: {name}")
            z.extractall(data_dir)
    print(f"[data] MD5 ok ({digest}); extracted to {out}")
    return out


@dataclass
class Dataset:
    doc_ids: list[str]                      # sorted lexicographically => index order == ID tie-break order
    titles: list[str]
    texts: list[str]
    query_ids: list[str]                    # sorted; row order of every score matrix
    query_texts: list[str]
    split_rows: dict[str, list[int]] = field(default_factory=dict)        # split -> query row indices
    qrels: dict[str, dict[int, dict[int, int]]] = field(default_factory=dict)  # split -> row -> {doc_idx: rel}
    dropped_qrels: int = 0

    @property
    def n_docs(self) -> int:
        return len(self.doc_ids)

    @property
    def n_queries(self) -> int:
        return len(self.query_ids)


def load_dataset(folder: Path) -> Dataset:
    docs = []
    with open(folder / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            docs.append((d["_id"], d.get("title", "") or "", d.get("text", "") or ""))
    docs.sort(key=lambda x: x[0])
    doc_ids = [d[0] for d in docs]
    doc_index = {d: i for i, d in enumerate(doc_ids)}

    queries = []
    with open(folder / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            queries.append((q["_id"], q.get("text", "") or ""))
    queries.sort(key=lambda x: x[0])
    query_ids = [q[0] for q in queries]
    q_index = {q: i for i, q in enumerate(query_ids)}

    ds = Dataset(doc_ids, [d[1] for d in docs], [d[2] for d in docs], query_ids, [q[1] for q in queries])
    for split in SPLITS:
        rel: dict[int, dict[int, int]] = {}
        with open(folder / "qrels" / f"{split}.tsv", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader)
            assert header[0] == "query-id", header
            for qid, did, score in reader:
                if qid not in q_index or did not in doc_index:
                    ds.dropped_qrels += 1
                    continue
                rel.setdefault(q_index[qid], {})[doc_index[did]] = int(score)
        ds.qrels[split] = rel
        ds.split_rows[split] = sorted(rel)
    return ds


def verify(ds: Dataset, cfg: dict) -> dict:
    """Recompute the structural facts in plan §3 and fail loudly if they changed."""
    exp = cfg["data"]["expected"]
    stats = {"corpus": ds.n_docs, "queries": ds.n_queries, "dropped_qrels": ds.dropped_qrels, "splits": {}}
    for split in SPLITS:
        labels = Counter(r for d in ds.qrels[split].values() for r in d.values())
        stats["splits"][split] = {
            "queries": len(ds.split_rows[split]),
            "qrels": sum(labels.values()),
            "labels": {str(k): v for k, v in sorted(labels.items())},
            "queries_without_positive": sum(1 for d in ds.qrels[split].values() if not any(r > 0 for r in d.values())),
        }
    problems = [f"corpus {ds.n_docs}!={exp['corpus']}" for _ in [0] if ds.n_docs != exp["corpus"]]
    problems += [f"queries {ds.n_queries}!={exp['queries']}" for _ in [0] if ds.n_queries != exp["queries"]]
    for split in SPLITS:
        n = stats["splits"][split]["queries"]
        if n != exp[split]:
            problems.append(f"{split} queries {n}!={exp[split]}")
    tr, dv, te = (set(ds.split_rows[s]) for s in SPLITS)
    overlap = (tr & dv) | (tr & te) | (dv & te)
    if overlap:
        problems.append(f"{len(overlap)} query IDs appear in more than one split")
    if problems:
        raise RuntimeError("dataset verification failed: " + "; ".join(problems))
    return stats


def duplicate_query_texts(ds: Dataset) -> dict:
    """Plan §3: normalised (strip + lower) query-text overlap between splits.

    Returns the overlap counts and the pre-registered filtered test subset
    (test queries whose normalised text also occurs in train).
    """
    norm = {s: {} for s in SPLITS}
    for s in SPLITS:
        for r in ds.split_rows[s]:
            norm[s].setdefault(ds.query_texts[r].strip().lower(), []).append(r)
    pairs = {}
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        pairs[f"{a}/{b}"] = len(set(norm[a]) & set(norm[b]))
    test_in_train = sorted(r for t, rows in norm["test"].items() if t in norm["train"] for r in rows)
    keep = [r for r in ds.split_rows["test"] if r not in set(test_in_train)]
    return {
        "shared_texts": pairs,
        "test_rows_overlapping_train": test_in_train,
        "test_query_ids_overlapping_train": [ds.query_ids[r] for r in test_in_train],
        "dedup_test_rows": keep,
    }
