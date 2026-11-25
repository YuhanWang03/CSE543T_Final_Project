"""Divergence-anchored pairwise objective with analytic gradient and Hessian (plan v3 §6–7).

    α_q   = σ(w·x_q)                     a_q = α_q(1 − α_q)
    m_qp  = ΔD + α_q c_qp,  c_qp = ΔB − ΔD
    J(w)  = (1/|Q|) Σ_q (1/n_q) Σ_p log(1 + e^{−m_qp})
            + (λ/2) ||w_{1:}||²  +  (γ/|Q|) Σ_q h_q (α_q − α_0)²

    t_q   = −(1/n_q) Σ_p σ(−m_qp) c_qp + 2γ h_q (α_q − α_0)
    ∇J    = (1/|Q|) Σ_q t_q a_q x_q + λ v,            v = (0, w_1, ..., w_{d−1})
    u_q   = (1/n_q) Σ_p σ(m_qp) σ(−m_qp) c_qp² + 2γ h_q
    ρ_q   = u_q a_q² + t_q a_q (1 − 2α_q)
    ∇²J   = (1/|Q|) Σ_q ρ_q x_q x_qᵀ + λ diag(0, 1, ..., 1)

All per-pair sums are aggregated per query with np.bincount, so memory stays O(#pairs + |Q| d).
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit


class Objective:
    def __init__(self, X: np.ndarray, q: np.ndarray, dB: np.ndarray, dD: np.ndarray,
                 alpha0: float, lam: float, gamma: float, h: np.ndarray | None = None):
        """X: [|Q|, d] standardised features with a leading bias column; q: pair -> query index 0..|Q|-1;
        dB, dD: standardised score differences (positive − negative) per pair."""
        self.X = np.asarray(X, dtype=np.float64)
        self.q = np.asarray(q, dtype=np.int64)
        self.dD = np.asarray(dD, dtype=np.float64)
        self.c = np.asarray(dB, dtype=np.float64) - self.dD
        self.nQ, self.d = self.X.shape
        n_q = np.bincount(self.q, minlength=self.nQ).astype(np.float64)
        if np.any(n_q == 0):
            raise ValueError("every query needs at least one pair")
        self.inv_n = 1.0 / n_q
        self.pair_w = self.inv_n[self.q] / self.nQ
        self.alpha0, self.lam, self.gamma = float(alpha0), float(lam), float(gamma)
        self.h = np.zeros(self.nQ) if h is None else np.asarray(h, dtype=np.float64)
        self.mask = np.ones(self.d)
        self.mask[0] = 0.0  # bias is not L2-penalised

    # -- shared pieces
    def _forward(self, w):
        a = expit(self.X @ w)
        m = self.dD + a[self.q] * self.c
        return a, m

    def _t(self, a, m):
        s = expit(-m)
        return (-np.bincount(self.q, weights=s * self.c, minlength=self.nQ) * self.inv_n
                + 2.0 * self.gamma * self.h * (a - self.alpha0))

    # -- public API
    def parts(self, w) -> dict:
        w = np.asarray(w, dtype=np.float64)
        a, m = self._forward(w)
        rank = float(np.sum(self.pair_w * np.logaddexp(0.0, -m)))
        l2 = 0.5 * self.lam * float(np.sum((w * self.mask) ** 2))
        anchor = self.gamma / self.nQ * float(np.sum(self.h * (a - self.alpha0) ** 2))
        return {"J": rank + l2 + anchor, "L_rank": rank, "l2": l2, "anchor": anchor}

    def value(self, w) -> float:
        return self.parts(w)["J"]

    def grad(self, w) -> np.ndarray:
        w = np.asarray(w, dtype=np.float64)
        a, m = self._forward(w)
        aa = a * (1.0 - a)
        return self.X.T @ (self._t(a, m) * aa) / self.nQ + self.lam * w * self.mask

    def value_and_grad(self, w):
        w = np.asarray(w, dtype=np.float64)
        a, m = self._forward(w)
        aa = a * (1.0 - a)
        J = (float(np.sum(self.pair_w * np.logaddexp(0.0, -m)))
             + 0.5 * self.lam * float(np.sum((w * self.mask) ** 2))
             + self.gamma / self.nQ * float(np.sum(self.h * (a - self.alpha0) ** 2)))
        g = self.X.T @ (self._t(a, m) * aa) / self.nQ + self.lam * w * self.mask
        return J, g

    def hess(self, w) -> np.ndarray:
        w = np.asarray(w, dtype=np.float64)
        a, m = self._forward(w)
        aa = a * (1.0 - a)
        t = self._t(a, m)
        sp = expit(m) * expit(-m)
        u = np.bincount(self.q, weights=sp * self.c ** 2, minlength=self.nQ) * self.inv_n + 2.0 * self.gamma * self.h
        rho = u * aa ** 2 + t * aa * (1.0 - 2.0 * a)
        return (self.X * rho[:, None]).T @ self.X / self.nQ + self.lam * np.diag(self.mask)

    def alphas(self, w) -> np.ndarray:
        return expit(self.X @ np.asarray(w, dtype=np.float64))


def anchor_weights(model: str, g: np.ndarray, tiny: float = 1e-8) -> tuple[np.ndarray | None, bool]:
    """h_q for R1–R4 from raw divergence g in [0,1]. Returns (h, degenerated_to_R1)."""
    g = np.asarray(g, dtype=np.float64)
    if model == "R1":
        return None, False
    if model == "R2":
        return np.ones_like(g), False
    num = g if model == "R3" else 1.0 - g if model == "R4" else None
    if num is None:
        raise ValueError(model)
    den = num.mean()
    if den < tiny:
        return np.zeros_like(g), True
    return num / den, False


def check_derivatives(obj: Objective, w: np.ndarray, steps=(1e-5, 1e-6), h_step: float = 1e-5) -> dict:
    """Central finite differences of J (for ∇J) and of ∇J (for ∇²J)."""
    w = np.asarray(w, dtype=np.float64)
    g = obj.grad(w)
    out = {"grad_inf": float(np.abs(g).max())}
    for s in steps:
        fd = np.array([(obj.value(w + s * e) - obj.value(w - s * e)) / (2 * s) for e in np.eye(len(w))])
        err = float(np.abs(fd - g).max())
        out[f"grad_abs_err_h{s:g}"] = err
        out[f"grad_rel_err_h{s:g}"] = err / max(float(np.abs(g).max()), 1e-12)
    H = obj.hess(w)
    Hfd = np.array([(obj.grad(w + h_step * e) - obj.grad(w - h_step * e)) / (2 * h_step) for e in np.eye(len(w))])
    Hfd = 0.5 * (Hfd + Hfd.T)
    out["hess_rel_err_fro"] = float(np.linalg.norm(H - Hfd) / max(np.linalg.norm(H), 1e-12))
    out["hess_abs_err_fro"] = float(np.linalg.norm(H - Hfd))
    out["hess_asym"] = float(np.abs(H - H.T).max())
    return out
