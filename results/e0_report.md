# E0 pre-diagnostics

Dev-best fixed α = 0.10

| split | oracle NDCG@10 | fixed NDCG@10 | Gap | frac |α*−α_fixed|>0.2 | frac queries with any gain | frac flat (all α equal) |
|---|---|---|---|---|---|---|
| train | 0.3927 | 0.3484 | 0.0443 | 0.175 | 0.490 | 0.308 |
| dev | 0.3743 | 0.3377 | 0.0365 | 0.154 | 0.491 | 0.309 |

**Decision (§4.1):** high: adaptive fusion is the main line (E3 retrieval comparison is the core result)

Spearman(feature, oracle α) on train:

- query_len: +0.046
- mean_idf: -0.119
- bm25_gap: -0.042
- dense_gap: +0.002
- bm25_top1_raw_log: +0.067
- query_coverage: +0.060
- dense_top10_std: +0.035
- g_jaccard: -0.194
- g_rbo: -0.164

Divergence distribution (train):

- g_jaccard: IQR=0.281, mode=1.000 (share 16.9%), distinct=11, median=0.824
- g_rbo: IQR=0.339, mode=1.000 (share 10.2%), distinct=1936, median=0.709

**Decision (§4.2):** use `g_jaccard` (mode-share trigger=False, IQR trigger=False)

Divergence `g_jaccard` vs |NDCG_BM25 − NDCG_Dense| (§4.3):

- train: Spearman=-0.140; mean |Δ| by g quartile = 0.088, 0.122, 0.125, 0.112
- dev: Spearman=-0.045; mean |Δ| by g quartile = 0.068, 0.118, 0.110, 0.142
