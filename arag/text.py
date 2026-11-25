"""Text preprocessing shared by both retrievers (plan v3 §3)."""
from __future__ import annotations

import re

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def bm25_tokenize(s: str) -> list[str]:
    """Lowercase + [a-z0-9]+ ; no stop words, no stemming."""
    return _TOKEN_RE.findall(s.lower())


def build_visible_texts(titles, texts, tokenizer, max_tokens: int, passage_prefix: str):
    """Cut each document to the prefix that fits in the E5 window.

    title and text are joined with a space. The budget leaves room for the special
    tokens and the "passage: " prefix, so "passage: " + visible <= max_tokens tokens.
    The cut is moved back to a word boundary so re-tokenisation cannot grow the length.
    Both BM25 and Dense use the returned visible text.
    """
    prefix_len = len(tokenizer(passage_prefix, add_special_tokens=False)["input_ids"])
    budget = max_tokens - tokenizer.num_special_tokens_to_add(pair=False) - prefix_len
    visible, full_len, truncated = [], [], []
    for title, text in zip(titles, texts):
        doc = f"{title} {text}".strip()
        enc = tokenizer(doc, add_special_tokens=False, return_offsets_mapping=True)
        n = len(enc["input_ids"])
        full_len.append(n)
        if n <= budget:
            visible.append(doc)
            truncated.append(False)
            continue
        word_ids = enc.word_ids()
        cut_tok = budget  # first excluded token
        while cut_tok > 0 and word_ids[cut_tok] is not None and word_ids[cut_tok] == word_ids[cut_tok - 1]:
            cut_tok -= 1  # move back to the start of the word that would be split
        vis = doc[: enc["offset_mapping"][cut_tok][0]].rstrip()
        # safety: re-check the final input length and back off if needed
        while len(tokenizer(passage_prefix + vis)["input_ids"]) > max_tokens:
            vis = vis[: vis.rstrip().rfind(" ")].rstrip()
        visible.append(vis)
        truncated.append(True)
    full_len = np.asarray(full_len)
    stats = {
        "token_budget_for_text": budget,
        "n_truncated": int(np.sum(truncated)),
        "frac_truncated": float(np.mean(truncated)),
        "full_token_length_percentiles": {str(p): float(np.percentile(full_len, p)) for p in (0, 10, 25, 50, 75, 90, 95, 99, 100)},
        "rule": "title + ' ' + text, cut to E5 token budget at a word boundary; BM25 and Dense share this text",
    }
    return visible, np.asarray(truncated), stats
