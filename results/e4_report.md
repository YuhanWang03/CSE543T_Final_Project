# E4 solver comparison and curvature

Common acceptance: ‖∇J‖∞ ≤ 1e-5 at the returned point; ≤ 2,000 iterations; ≤ 10,000 f/g calls. Time = median of 3 runs for the anchor start (single run otherwise), history evaluation excluded.

## R1 (λ=0.001, γ=0)

| start | method | converged | J | ‖∇J‖∞ | iters | f / g / H calls | time s | λ_min(∇²J) | κ | dev NDCG@10 | stop reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| anchor | GD | **no** | 0.748184857 | 6.7e-05 | 2000 | 2001 / 2001 / 0 | 79.981 | 1.27e-03 | 21.2 | 0.3352 | max_iterations |
| anchor | BFGS | yes | 0.748181847 | 9.4e-06 | 50 | 55 / 55 / 0 | 2.054 | 1.22e-03 | 21.9 | 0.3349 | Optimization terminated successfully. |
| anchor | L-BFGS-B | yes | 0.748181845 | 4.4e-06 | 11 | 13 / 13 / 0 | 0.493 | 1.22e-03 | 21.9 | 0.3349 | CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL |
| anchor | Newton-CG | yes | 0.748181843 | 1.4e-07 | 8 | 9 / 9 / 8 | 0.912 | 1.22e-03 | 21.9 | 0.3349 | Optimization terminated successfully. |
| anchor | trust-exact | yes | 0.748181843 | 3.8e-07 | 6 | 7 / 7 / 7 | 0.763 | 1.22e-03 | 21.9 | 0.3349 | Optimization terminated successfully. |
| perturbed_s11 | GD | **no** | 0.748185167 | 7.1e-05 | 2000 | 2001 / 2001 / 0 | 83.613 | 1.27e-03 | 21.2 | 0.3352 | max_iterations |
| perturbed_s11 | BFGS | yes | 0.748181843 | 1.9e-06 | 47 | 53 / 53 / 0 | 2.081 | 1.22e-03 | 21.9 | 0.3349 | Optimization terminated successfully. |
| perturbed_s11 | L-BFGS-B | yes | 0.748181847 | 7.1e-06 | 11 | 14 / 14 / 0 | 0.555 | 1.22e-03 | 21.9 | 0.3349 | CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL |
| perturbed_s11 | Newton-CG | yes | 0.748181843 | 1.5e-07 | 8 | 9 / 9 / 8 | 0.956 | 1.22e-03 | 21.9 | 0.3349 | Optimization terminated successfully. |
| perturbed_s11 | trust-exact | yes | 0.748181843 | 5.6e-07 | 6 | 7 / 7 / 7 | 0.800 | 1.22e-03 | 21.9 | 0.3349 | Optimization terminated successfully. |
| perturbed_s29 | GD | **no** | 0.748184814 | 6.7e-05 | 2000 | 2001 / 2001 / 0 | 138.332 | 1.27e-03 | 21.2 | 0.3352 | max_iterations |
| perturbed_s29 | BFGS | yes | 0.748181844 | 6.0e-06 | 51 | 55 / 55 / 0 | 6.890 | 1.22e-03 | 21.9 | 0.3349 | Optimization terminated successfully. |
| perturbed_s29 | L-BFGS-B | yes | 0.748181844 | 2.7e-06 | 12 | 14 / 14 / 0 | 1.760 | 1.22e-03 | 21.9 | 0.3349 | CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL |
| perturbed_s29 | Newton-CG | yes | 0.748181843 | 8.1e-08 | 8 | 9 / 9 / 8 | 3.348 | 1.22e-03 | 21.9 | 0.3349 | Optimization terminated successfully. |
| perturbed_s29 | trust-exact | yes | 0.748181843 | 3.2e-07 | 6 | 7 / 7 / 7 | 2.816 | 1.22e-03 | 21.9 | 0.3349 | Optimization terminated successfully. |

- Hessian at the solution: λ_min=1.220e-03, λ_max=2.671e-02, κ=21.9.
- GD (Armijo accepted t=1, i.e. fixed step 1): predicted local rate 1−λ_min = 0.99878 → ≈6286 iterations to reach the tolerance; observed 2000 (converged=False). With an ideally scaled step the rate would be (κ−1)/(κ+1) = 0.913 → ≈84 iterations.
- Hessian at the anchor start indefinite: True; fraction of random points with negative curvature: N(0,0.5^2): 95%, N(0,1.0^2): 94%, N(0,3.0^2): 90%.
- Multi-start: 12/15 runs converged; spread of J among them 4.65e-09; max ‖w − w_BFGS,anchor‖ = 2.34e-03; all positive definite: True.

## R3 (λ=0.001, γ=1)

| start | method | converged | J | ‖∇J‖∞ | iters | f / g / H calls | time s | λ_min(∇²J) | κ | dev NDCG@10 | stop reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| anchor | GD | yes | 0.778413977 | 1.0e-05 | 334 | 335 / 335 / 0 | 44.564 | 1.72e-02 | 9.79 | 0.3357 | gradient_tolerance |
| anchor | BFGS | yes | 0.778413973 | 7.8e-06 | 29 | 32 / 32 / 0 | 3.937 | 1.72e-02 | 9.79 | 0.3357 | Optimization terminated successfully. |
| anchor | L-BFGS-B | yes | 0.778413972 | 4.1e-06 | 9 | 11 / 11 / 0 | 1.381 | 1.72e-02 | 9.78 | 0.3357 | CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL |
| anchor | Newton-CG | yes | 0.778413971 | 2.9e-08 | 6 | 7 / 7 / 6 | 2.350 | 1.72e-02 | 9.78 | 0.3357 | Optimization terminated successfully. |
| anchor | trust-exact | yes | 0.778413971 | 5.1e-07 | 5 | 6 / 6 / 6 | 2.413 | 1.72e-02 | 9.78 | 0.3357 | Optimization terminated successfully. |
| perturbed_s11 | GD | yes | 0.778413977 | 9.9e-06 | 251 | 252 / 252 / 0 | 32.619 | 1.72e-02 | 9.79 | 0.3357 | gradient_tolerance |
| perturbed_s11 | BFGS | yes | 0.778413972 | 5.8e-06 | 28 | 32 / 32 / 0 | 1.943 | 1.72e-02 | 9.79 | 0.3357 | Optimization terminated successfully. |
| perturbed_s11 | L-BFGS-B | yes | 0.778413971 | 1.5e-06 | 10 | 12 / 12 / 0 | 0.468 | 1.72e-02 | 9.78 | 0.3357 | CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL |
| perturbed_s11 | Newton-CG | yes | 0.778413971 | 6.0e-08 | 6 | 7 / 7 / 6 | 0.714 | 1.72e-02 | 9.78 | 0.3357 | Optimization terminated successfully. |
| perturbed_s11 | trust-exact | yes | 0.778413971 | 1.6e-07 | 5 | 6 / 6 / 6 | 0.678 | 1.72e-02 | 9.78 | 0.3357 | Optimization terminated successfully. |
| perturbed_s29 | GD | yes | 0.778413977 | 9.9e-06 | 336 | 337 / 337 / 0 | 13.163 | 1.72e-02 | 9.79 | 0.3357 | gradient_tolerance |
| perturbed_s29 | BFGS | yes | 0.778413973 | 7.8e-06 | 30 | 33 / 33 / 0 | 1.180 | 1.72e-02 | 9.79 | 0.3357 | Optimization terminated successfully. |
| perturbed_s29 | L-BFGS-B | yes | 0.778413972 | 4.4e-06 | 8 | 11 / 11 / 0 | 0.399 | 1.72e-02 | 9.78 | 0.3357 | CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL |
| perturbed_s29 | Newton-CG | yes | 0.778413971 | 1.2e-08 | 6 | 8 / 8 / 6 | 0.701 | 1.72e-02 | 9.78 | 0.3357 | Optimization terminated successfully. |
| perturbed_s29 | trust-exact | yes | 0.778413971 | 2.2e-06 | 5 | 6 / 6 / 6 | 0.628 | 1.72e-02 | 9.78 | 0.3357 | Optimization terminated successfully. |

- Hessian at the solution: λ_min=1.721e-02, λ_max=1.685e-01, κ=9.79.
- GD (Armijo accepted t=1, i.e. fixed step 1): predicted local rate 1−λ_min = 0.98279 → ≈442 iterations to reach the tolerance; observed 334 (converged=True). With an ideally scaled step the rate would be (κ−1)/(κ+1) = 0.815 → ≈37 iterations.
- Hessian at the anchor start indefinite: False; fraction of random points with negative curvature: N(0,0.5^2): 74%, N(0,1.0^2): 96%, N(0,3.0^2): 100%.
- Multi-start: 15/15 runs converged; spread of J among them 6.00e-09; max ‖w − w_BFGS,anchor‖ = 8.78e-04; all positive definite: True.

## κ at the E3 solutions vs γ

| model | λ | γ | λ_min | λ_max | κ | BFGS iterations |
|---|---|---|---|---|---|---|
| R1 | 1e-04 | 0 | 2.57e-04 | 1.60e-02 | 62.3 | 56 |
| R1 | 1e-03 | 0 | 1.22e-03 | 2.67e-02 | 21.9 | 50 |
| R1 | 1e-02 | 0 | 1.27e-02 | 5.20e-02 | 4.08 | 24 |
| R2 | 1e-04 | 0.001 | 2.58e-04 | 1.62e-02 | 63 | 57 |
| R2 | 1e-04 | 0.01 | 2.67e-04 | 1.85e-02 | 69.4 | 54 |
| R2 | 1e-04 | 0.1 | 3.67e-04 | 4.86e-02 | 132 | 45 |
| R2 | 1e-04 | 1 | 1.66e-02 | 1.57e-01 | 9.47 | 27 |
| R2 | 1e-03 | 0.001 | 1.22e-03 | 2.69e-02 | 22 | 51 |
| R2 | 1e-03 | 0.01 | 1.25e-03 | 2.85e-02 | 22.9 | 45 |
| R2 | 1e-03 | 0.1 | 2.04e-03 | 5.33e-02 | 26.2 | 43 |
| R2 | 1e-03 | 1 | 1.76e-02 | 1.58e-01 | 8.97 | 27 |
| R2 | 1e-02 | 0.001 | 1.27e-02 | 5.21e-02 | 4.09 | 24 |
| R2 | 1e-02 | 0.01 | 1.29e-02 | 5.36e-02 | 4.17 | 24 |
| R2 | 1e-02 | 0.1 | 1.41e-02 | 7.22e-02 | 5.12 | 29 |
| R2 | 1e-02 | 1 | 2.72e-02 | 1.62e-01 | 5.96 | 24 |
| R3 | 1e-04 | 0.001 | 2.58e-04 | 1.63e-02 | 63.2 | 57 |
| R3 | 1e-04 | 0.01 | 2.67e-04 | 1.89e-02 | 70.7 | 53 |
| R3 | 1e-04 | 0.1 | 4.57e-04 | 5.33e-02 | 117 | 47 |
| R3 | 1e-04 | 1 | 1.62e-02 | 1.68e-01 | 10.4 | 29 |
| R3 | 1e-03 | 0.001 | 1.22e-03 | 2.69e-02 | 22 | 51 |
| R3 | 1e-03 | 0.01 | 1.26e-03 | 2.87e-02 | 22.9 | 45 |
| R3 | 1e-03 | 0.1 | 2.13e-03 | 5.79e-02 | 27.2 | 42 |
| R3 | 1e-03 | 1 | 1.72e-02 | 1.68e-01 | 9.79 | 29 |
| R3 | 1e-02 | 0.001 | 1.27e-02 | 5.22e-02 | 4.09 | 24 |
| R3 | 1e-02 | 0.01 | 1.29e-02 | 5.39e-02 | 4.19 | 24 |
| R3 | 1e-02 | 0.1 | 1.42e-02 | 7.60e-02 | 5.37 | 28 |
| R3 | 1e-02 | 1 | 2.69e-02 | 1.73e-01 | 6.44 | 23 |
| R4 | 1e-04 | 0.001 | 2.58e-04 | 1.61e-02 | 62.6 | 56 |
| R4 | 1e-04 | 0.01 | 2.65e-04 | 1.72e-02 | 64.9 | 53 |
| R4 | 1e-04 | 0.1 | 2.88e-04 | 2.86e-02 | 99.5 | 48 |
| R4 | 1e-04 | 1 | 1.28e-02 | 9.79e-02 | 7.64 | 24 |
| R4 | 1e-03 | 0.001 | 1.22e-03 | 2.68e-02 | 22 | 51 |
| R4 | 1e-03 | 0.01 | 1.22e-03 | 2.77e-02 | 22.7 | 51 |
| R4 | 1e-03 | 0.1 | 1.44e-03 | 3.69e-02 | 25.6 | 43 |
| R4 | 1e-03 | 1 | 1.41e-02 | 9.87e-02 | 7 | 24 |
| R4 | 1e-02 | 0.001 | 1.27e-02 | 5.20e-02 | 4.09 | 24 |
| R4 | 1e-02 | 0.01 | 1.28e-02 | 5.26e-02 | 4.11 | 25 |
| R4 | 1e-02 | 0.1 | 1.37e-02 | 5.85e-02 | 4.28 | 26 |
| R4 | 1e-02 | 1 | 2.49e-02 | 1.03e-01 | 4.15 | 29 |
