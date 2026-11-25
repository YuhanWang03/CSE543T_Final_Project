"""Solver suite for E4 with common accounting (plan v3 §8).

Every solver gets the same analytic derivatives, the same start and the same acceptance test
(||∇J||_inf <= 1e-5 at the returned point). Library stopping rules are set explicitly; whatever a
library reports, the final gradient is re-checked with the common criterion.

Timing: wall time of objective / gradient / Hessian evaluations, line searches and solver internals.
The per-iteration history callback evaluates J for plotting; that evaluation is excluded from both
the timer and the call counters.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize

from .objective import Objective

GTOL = 1e-5
MAXITER = 2000
MAXFUN = 10_000
METHODS = ("GD", "BFGS", "L-BFGS-B", "Newton-CG", "trust-exact")


class Counted:
    """Wraps an Objective, counting f / g / H evaluations."""

    def __init__(self, obj: Objective):
        self.obj = obj
        self.nf = self.ng = self.nh = 0

    def fg(self, w):
        self.nf += 1
        self.ng += 1
        return self.obj.value_and_grad(w)

    def f(self, w):
        self.nf += 1
        return self.obj.value(w)

    def g(self, w):
        self.ng += 1
        return self.obj.grad(w)

    def h(self, w):
        self.nh += 1
        return self.obj.hess(w)


class History:
    def __init__(self, obj: Objective):
        self.obj = obj
        self.t0 = time.perf_counter()
        self.overhead = 0.0
        self.rows = []  # (iteration, J, ||g||inf, elapsed_excluding_overhead)

    def record(self, w):
        t = time.perf_counter()
        J, g = self.obj.value_and_grad(w)
        self.rows.append((len(self.rows), float(J), float(np.abs(g).max()), t - self.t0 - self.overhead))
        self.overhead += time.perf_counter() - t

    def elapsed(self):
        return time.perf_counter() - self.t0 - self.overhead


def _gd_armijo(c: Counted, w0, hist: History, t0=1.0, shrink=0.5, c1=1e-4):
    w = np.array(w0, dtype=float)
    f, g = c.fg(w)
    status, nit = "max_iterations", 0
    for nit in range(1, MAXITER + 1):
        if np.abs(g).max() <= GTOL:
            nit -= 1
            status = "gradient_tolerance"
            break
        t, gg = t0, float(g @ g)
        while True:
            w_new = w - t * g
            f_new = c.f(w_new)
            if f_new <= f - c1 * t * gg:
                break
            t *= shrink
            if t < 1e-20 or c.nf >= MAXFUN:
                break
        if f_new > f - c1 * t * gg:
            status = "line_search_failed" if t < 1e-20 else "max_function_calls"
            break
        w = w_new
        f, g = f_new, c.g(w)
        hist.record(w)
        if c.nf >= MAXFUN:
            status = "max_function_calls"
            break
    return w, nit, status


def run_solver(obj: Objective, w0: np.ndarray, method: str) -> dict:
    c = Counted(obj)
    hist = History(obj)
    hist.record(np.asarray(w0, dtype=float))  # iteration 0 (its cost is excluded)
    cb = lambda xk, *args: hist.record(xk)
    if method == "GD":
        w, nit, status = _gd_armijo(c, w0, hist)
        lib_success, msg = status == "gradient_tolerance", status
    else:
        if method == "BFGS":
            r = minimize(c.fg, w0, jac=True, method="BFGS", callback=cb,
                         options={"gtol": GTOL, "norm": np.inf, "maxiter": MAXITER})
        elif method == "L-BFGS-B":
            # no bounds; pgtol = inf-norm of the gradient; ftol=0 so only gtol / budgets stop it
            r = minimize(c.fg, w0, jac=True, method="L-BFGS-B", callback=cb,
                         options={"gtol": GTOL, "ftol": 0.0, "maxcor": 10, "maxiter": MAXITER, "maxfun": MAXFUN})
        elif method == "Newton-CG":
            # Newton-CG has no gradient tolerance; a tiny xtol lets the common criterion decide
            r = minimize(c.fg, w0, jac=True, hess=c.h, method="Newton-CG", callback=cb,
                         options={"xtol": 1e-12, "maxiter": MAXITER})
        elif method == "trust-exact":
            # gtol is on the 2-norm, which bounds the inf-norm from above
            r = minimize(c.fg, w0, jac=True, hess=c.h, method="trust-exact", callback=cb,
                         options={"gtol": GTOL, "maxiter": MAXITER})
        else:
            raise ValueError(method)
        w, nit, lib_success, msg = r.x, int(r.nit), bool(r.success), str(r.message)
    elapsed = hist.elapsed()
    g = obj.grad(w)
    ev = np.linalg.eigvalsh(obj.hess(w))
    return {
        "method": method, "w": [float(x) for x in w], "time_s": elapsed, "nit": nit,
        "n_f": c.nf, "n_g": c.ng, "n_h": c.nh, "library_success": lib_success, "message": msg,
        "grad_inf": float(np.abs(g).max()), "converged": bool(np.abs(g).max() <= GTOL),
        **obj.parts(w),
        "hess_eig": [float(x) for x in ev], "hess_eig_min": float(ev.min()), "hess_eig_max": float(ev.max()),
        "cond": float(ev.max() / ev.min()) if ev.min() > 0 else float("inf"),
        "history": hist.rows,
    }
