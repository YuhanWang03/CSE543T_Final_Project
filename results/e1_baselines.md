| System | train NDCG@10 | train R@10 | train MAP@10 | dev NDCG@10 | dev R@10 | dev MAP@10 |
|---|---|---|---|---|---|---|
| BM25 | 0.3029 | 0.1338 | 0.1024 | 0.2662 | 0.1097 | 0.0787 |
| Dense (E5-small-v2) | 0.3258 | 0.1443 | 0.1071 | 0.3221 | 0.1428 | 0.1033 |
| Fixed fusion α=0.50 | 0.3356 | 0.1531 | 0.1146 | 0.3080 | 0.1343 | 0.0949 |
| Fixed fusion α=0.10 (dev-best) | 0.3484 | 0.1582 | 0.1184 | 0.3377 | 0.1507 | 0.1083 |
| RRF k=60 | 0.3281 | 0.1490 | 0.1092 | 0.3053 | 0.1336 | 0.0926 |
| RRF k=20 (dev-selected) | 0.3360 | 0.1530 | 0.1126 | 0.3152 | 0.1419 | 0.0977 |

Test split intentionally not evaluated (locked until configuration freeze).
