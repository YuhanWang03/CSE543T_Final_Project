# Test evaluation (locked configuration; 323 queries, dedup subset 319)

| system | NDCG@10 | R@10 | MAP@10 | NDCG@10 (dedup) | Δ vs fixed | improved | severe (<−0.05) | Δ p10 |
|---|---|---|---|---|---|---|---|---|
| BM25 | 0.3075 | 0.1524 | 0.1169 | 0.3074 | -0.0401 | 0.195 | 0.294 | -0.226 |
| Dense | 0.3256 | 0.1596 | 0.1189 | 0.3259 | -0.0220 | 0.198 | 0.214 | -0.123 |
| Fixed α=0.50 | 0.3364 | 0.1612 | 0.1295 | 0.3364 | -0.0112 | 0.229 | 0.204 | -0.118 |
| Fixed α=0.10 (dev-best) | 0.3476 | 0.1715 | 0.1340 | 0.3476 | +0.0000 | 0.000 | 0.000 | +0.000 |
| RRF k=60 | 0.3360 | 0.1602 | 0.1273 | 0.3358 | -0.0116 | 0.223 | 0.183 | -0.094 |
| RRF k=20 (dev-selected) | 0.3420 | 0.1656 | 0.1310 | 0.3417 | -0.0056 | 0.232 | 0.155 | -0.078 |
| R1 | 0.3441 | 0.1680 | 0.1314 | 0.3445 | -0.0036 | 0.198 | 0.108 | -0.051 |
| R2 | 0.3508 | 0.1730 | 0.1365 | 0.3506 | +0.0031 | 0.142 | 0.028 | -0.006 |
| R3 | 0.3509 | 0.1730 | 0.1366 | 0.3508 | +0.0032 | 0.149 | 0.031 | -0.006 |
| R4 | 0.3465 | 0.1692 | 0.1324 | 0.3469 | -0.0012 | 0.189 | 0.087 | -0.040 |

Paired bootstrap on test NDCG@10 (10,000 resamples of queries; 95% percentile CI):

- R3 − Fixed α=0.10 (dev-best): mean Δ = +0.0032, 95% CI [+0.0000, +0.0065] **(primary, pre-registered)**
- R1 − Fixed α=0.10 (dev-best): mean Δ = -0.0036, 95% CI [-0.0089, +0.0017]
- R2 − Fixed α=0.10 (dev-best): mean Δ = +0.0031, 95% CI [+0.0000, +0.0063]
- R4 − Fixed α=0.10 (dev-best): mean Δ = -0.0012, 95% CI [-0.0056, +0.0034]
- R3 − R2: mean Δ = +0.0001, 95% CI [-0.0006, +0.0008]
- R3 − R1: mean Δ = +0.0068, 95% CI [+0.0011, +0.0128]

Weak-negative resampling (E6), test NDCG@10:

- R1: mean 0.3438, sd 0.0003; s17=0.3441, s23=0.3432, s42=0.3438, s71=0.3439, s101=0.3439
- R2: mean 0.3508, sd 0.0002; s17=0.3508, s23=0.3506, s42=0.3508, s71=0.3510, s101=0.3508
- R3: mean 0.3508, sd 0.0001; s17=0.3509, s23=0.3508, s42=0.3508, s71=0.3507, s101=0.3508

E5 variants, test NDCG@10:

- R3 − g feature (anchor kept): 0.3509
- R3 − anchor term (g feature kept): 0.3441
- R1 uniform negatives: 0.3486
- R2 uniform negatives: 0.3536
- R3 uniform negatives: 0.3531
- R3 no head skipping (v2 hard set): 0.3487

Cases (R3 vs fixed fusion):

- [best] PLAIN-3342 Δ=+0.182 (R3 0.626, fixed 0.444, BM25 0.636, Dense 0.148); α_R3=0.12, g=0.82, |rel|=3: “Is Coconut Milk Good For You?”
- [best] PLAIN-1601 Δ=+0.133 (R3 0.697, fixed 0.564, BM25 0.581, Dense 0.542); α_R3=0.19, g=0.67, |rel|=74: “memory”
- [best] PLAIN-2291 Δ=+0.125 (R3 0.485, fixed 0.360, BM25 0.452, Dense 0.209); α_R3=0.15, g=0.82, |rel|=5: “ultra-processed foods”
- [worst] PLAIN-3241 Δ=-0.155 (R3 0.000, fixed 0.155, BM25 0.000, Dense 0.421); α_R3=0.24, g=0.95, |rel|=29: “Apthous Ulcer Mystery Solved”
- [worst] PLAIN-91 Δ=-0.093 (R3 0.281, fixed 0.373, BM25 0.177, Dense 0.380); α_R3=0.14, g=0.75, |rel|=64: “Chronic Headaches and Pork Parasites”
- [worst] PLAIN-3312 Δ=-0.093 (R3 0.188, fixed 0.280, BM25 0.124, Dense 0.248); α_R3=0.18, g=0.67, |rel|=13: “Sexually Transmitted Fish Toxin”

Cost per query: BM25 3.75 ms, dense query encoding + scoring 0.32 ms, fusion + ranking 0.407 ms; one-off document encoding 28.1 s; training (BFGS) R1 2.04s, R2 1.09s, R3 1.15s, R4 1.81s.
