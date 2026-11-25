Feature set: `F5`; unique fits: 39; not converged (‖∇J‖∞ > 1e-5): 0

| model | λ | γ | dev NDCG@10 | R@10 | MAP@10 | Δ vs fixed | improved | severe (<−0.05) | Δ p10 | α mean (sd) | ‖∇J‖∞ | nit | time s | κ(∇²J) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R1 | 1e-03 | 0 | 0.3349 | 0.1478 | 0.1086 | -0.0029 | 0.216 | 0.136 | -0.067 | 0.319 (0.344) | 9.4e-06 | 50 | 2.04 | 21.9 |
| R2 | 1e-04 | 1 | 0.3359 | 0.1479 | 0.1065 | -0.0018 | 0.154 | 0.080 | -0.026 | 0.186 (0.074) | 9.4e-06 | 27 | 1.09 | 9.47 |
| R3 | 1e-03 | 1 | 0.3357 | 0.1479 | 0.1065 | -0.0020 | 0.164 | 0.083 | -0.030 | 0.185 (0.067) | 7.8e-06 | 29 | 1.15 | 9.79 |
| R4 | 1e-03 | 0.1 | 0.3361 | 0.1482 | 0.1086 | -0.0016 | 0.207 | 0.108 | -0.060 | 0.290 (0.303) | 1.6e-06 | 43 | 1.81 | 25.6 |

Reference: dev-best fixed fusion α=0.10, dev NDCG@10 = 0.3377

All runs (dev NDCG@10):

- R1 λ=1e-04 γ=0: dev NDCG@10=0.3318  Δfixed=-0.0059  severe=0.157  |∇J|∞=7.4e-06  conv=True  nit=56  t=2.27s  κ=62.3
- R1 λ=1e-03 γ=0: dev NDCG@10=0.3349  Δfixed=-0.0029  severe=0.136  |∇J|∞=9.4e-06  conv=True  nit=50  t=2.04s  κ=21.9
- R1 λ=1e-02 γ=0: dev NDCG@10=0.3320  Δfixed=-0.0057  severe=0.123  |∇J|∞=7.9e-06  conv=True  nit=24  t=1.14s  κ=4.08
- R2 λ=1e-04 γ=0.001: dev NDCG@10=0.3318  Δfixed=-0.0059  severe=0.157  |∇J|∞=6.5e-06  conv=True  nit=57  t=2.32s  κ=63
- R2 λ=1e-04 γ=0.01: dev NDCG@10=0.3322  Δfixed=-0.0055  severe=0.154  |∇J|∞=4.2e-06  conv=True  nit=54  t=2.22s  κ=69.4
- R2 λ=1e-04 γ=0.1: dev NDCG@10=0.3358  Δfixed=-0.0019  severe=0.130  |∇J|∞=6.1e-06  conv=True  nit=45  t=1.85s  κ=132
- R2 λ=1e-04 γ=1: dev NDCG@10=0.3359  Δfixed=-0.0018  severe=0.080  |∇J|∞=9.4e-06  conv=True  nit=27  t=1.09s  κ=9.47
- R2 λ=1e-03 γ=0.001: dev NDCG@10=0.3348  Δfixed=-0.0029  severe=0.136  |∇J|∞=4.8e-06  conv=True  nit=51  t=2.07s  κ=22
- R2 λ=1e-03 γ=0.01: dev NDCG@10=0.3352  Δfixed=-0.0025  severe=0.130  |∇J|∞=7.7e-06  conv=True  nit=45  t=1.90s  κ=22.9
- R2 λ=1e-03 γ=0.1: dev NDCG@10=0.3355  Δfixed=-0.0022  severe=0.114  |∇J|∞=6.7e-06  conv=True  nit=43  t=1.73s  κ=26.2
- R2 λ=1e-03 γ=1: dev NDCG@10=0.3357  Δfixed=-0.0020  severe=0.083  |∇J|∞=9.4e-06  conv=True  nit=27  t=1.09s  κ=8.97
- R2 λ=1e-02 γ=0.001: dev NDCG@10=0.3320  Δfixed=-0.0057  severe=0.123  |∇J|∞=6.6e-06  conv=True  nit=24  t=1.14s  κ=4.09
- R2 λ=1e-02 γ=0.01: dev NDCG@10=0.3323  Δfixed=-0.0054  severe=0.123  |∇J|∞=7.0e-06  conv=True  nit=24  t=1.14s  κ=4.17
- R2 λ=1e-02 γ=0.1: dev NDCG@10=0.3340  Δfixed=-0.0037  severe=0.105  |∇J|∞=7.9e-06  conv=True  nit=29  t=1.24s  κ=5.12
- R2 λ=1e-02 γ=1: dev NDCG@10=0.3356  Δfixed=-0.0021  severe=0.086  |∇J|∞=2.6e-06  conv=True  nit=24  t=0.97s  κ=5.96
- R3 λ=1e-04 γ=0.001: dev NDCG@10=0.3317  Δfixed=-0.0060  severe=0.157  |∇J|∞=1.5e-06  conv=True  nit=57  t=2.30s  κ=63.2
- R3 λ=1e-04 γ=0.01: dev NDCG@10=0.3323  Δfixed=-0.0054  severe=0.154  |∇J|∞=8.7e-06  conv=True  nit=53  t=2.18s  κ=70.7
- R3 λ=1e-04 γ=0.1: dev NDCG@10=0.3355  Δfixed=-0.0023  severe=0.130  |∇J|∞=2.5e-06  conv=True  nit=47  t=1.87s  κ=117
- R3 λ=1e-04 γ=1: dev NDCG@10=0.3357  Δfixed=-0.0020  severe=0.083  |∇J|∞=8.9e-06  conv=True  nit=29  t=1.15s  κ=10.4
- R3 λ=1e-03 γ=0.001: dev NDCG@10=0.3348  Δfixed=-0.0029  severe=0.136  |∇J|∞=4.6e-06  conv=True  nit=51  t=2.07s  κ=22
- R3 λ=1e-03 γ=0.01: dev NDCG@10=0.3348  Δfixed=-0.0029  severe=0.130  |∇J|∞=9.9e-06  conv=True  nit=45  t=1.88s  κ=22.9
- R3 λ=1e-03 γ=0.1: dev NDCG@10=0.3354  Δfixed=-0.0023  severe=0.117  |∇J|∞=9.7e-06  conv=True  nit=42  t=1.69s  κ=27.2
- R3 λ=1e-03 γ=1: dev NDCG@10=0.3357  Δfixed=-0.0020  severe=0.083  |∇J|∞=7.8e-06  conv=True  nit=29  t=1.15s  κ=9.79
- R3 λ=1e-02 γ=0.001: dev NDCG@10=0.3320  Δfixed=-0.0057  severe=0.123  |∇J|∞=6.4e-06  conv=True  nit=24  t=1.15s  κ=4.09
- R3 λ=1e-02 γ=0.01: dev NDCG@10=0.3325  Δfixed=-0.0052  severe=0.123  |∇J|∞=8.4e-06  conv=True  nit=24  t=1.13s  κ=4.19
- R3 λ=1e-02 γ=0.1: dev NDCG@10=0.3343  Δfixed=-0.0035  severe=0.105  |∇J|∞=7.6e-06  conv=True  nit=28  t=1.24s  κ=5.37
- R3 λ=1e-02 γ=1: dev NDCG@10=0.3351  Δfixed=-0.0026  severe=0.086  |∇J|∞=6.7e-06  conv=True  nit=23  t=0.94s  κ=6.44
- R4 λ=1e-04 γ=0.001: dev NDCG@10=0.3318  Δfixed=-0.0059  severe=0.157  |∇J|∞=6.3e-06  conv=True  nit=56  t=2.27s  κ=62.6
- R4 λ=1e-04 γ=0.01: dev NDCG@10=0.3320  Δfixed=-0.0058  severe=0.157  |∇J|∞=5.3e-06  conv=True  nit=53  t=2.20s  κ=64.9
- R4 λ=1e-04 γ=0.1: dev NDCG@10=0.3324  Δfixed=-0.0053  severe=0.151  |∇J|∞=7.8e-06  conv=True  nit=48  t=2.01s  κ=99.5
- R4 λ=1e-04 γ=1: dev NDCG@10=0.3343  Δfixed=-0.0035  severe=0.090  |∇J|∞=6.4e-06  conv=True  nit=24  t=1.09s  κ=7.64
- R4 λ=1e-03 γ=0.001: dev NDCG@10=0.3348  Δfixed=-0.0029  severe=0.136  |∇J|∞=4.1e-06  conv=True  nit=51  t=2.08s  κ=22
- R4 λ=1e-03 γ=0.01: dev NDCG@10=0.3352  Δfixed=-0.0025  severe=0.130  |∇J|∞=4.6e-06  conv=True  nit=51  t=2.09s  κ=22.7
- R4 λ=1e-03 γ=0.1: dev NDCG@10=0.3361  Δfixed=-0.0016  severe=0.108  |∇J|∞=1.6e-06  conv=True  nit=43  t=1.81s  κ=25.6
- R4 λ=1e-03 γ=1: dev NDCG@10=0.3347  Δfixed=-0.0030  severe=0.086  |∇J|∞=2.4e-06  conv=True  nit=24  t=1.05s  κ=7
- R4 λ=1e-02 γ=0.001: dev NDCG@10=0.3320  Δfixed=-0.0057  severe=0.123  |∇J|∞=8.0e-06  conv=True  nit=24  t=1.14s  κ=4.09
- R4 λ=1e-02 γ=0.01: dev NDCG@10=0.3324  Δfixed=-0.0053  severe=0.120  |∇J|∞=5.3e-06  conv=True  nit=25  t=1.18s  κ=4.11
- R4 λ=1e-02 γ=0.1: dev NDCG@10=0.3330  Δfixed=-0.0048  severe=0.108  |∇J|∞=9.7e-06  conv=True  nit=26  t=1.18s  κ=4.28
- R4 λ=1e-02 γ=1: dev NDCG@10=0.3349  Δfixed=-0.0028  severe=0.086  |∇J|∞=8.8e-06  conv=True  nit=29  t=1.16s  κ=4.15
