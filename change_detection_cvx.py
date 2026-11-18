"""Python/CVXPY port of CVX.m: change detection in a time-varying AR(3) model.

    y(t+3) = a(t) y(t+2) + b(t) y(t+1) + c(t) y(t) + v(t)

The coefficients are piecewise constant, so we fit them by least squares plus a
total-variation penalty (the convex surrogate of counting change points), with box
constraints from the known coefficient ranges.

Usage:  python change_detection_cvx.py [--lam 4.3] [--save figures/cvx_estimate.png]
"""
import argparse
import os

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


def solve(y, lam):
    T = len(y)
    a, b, c = cp.Variable(T - 3), cp.Variable(T - 3), cp.Variable(T - 3)
    pred = cp.multiply(a, y[2:T - 1]) + cp.multiply(b, y[1:T - 2]) + cp.multiply(c, y[0:T - 3])
    fit = cp.sum_squares(y[3:T] - pred)
    tv = sum(lam * cp.norm1(cp.diff(v)) for v in (a, b, c))
    cons = [a >= -0.5, a <= 1, b >= -0.8, b <= -0.2, c >= -0.2, c <= 0.2]
    cp.Problem(cp.Minimize(fit + tv), cons).solve()
    return a.value, b.value, c.value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=4.3)
    ap.add_argument("--save", default=None, help="save the figure instead of showing it")
    args = ap.parse_args()

    data = loadmat("change_detection.mat")
    y = data["y"].ravel()
    true = {k: data[k].ravel() for k in "abc"}

    est = dict(zip("abc", solve(y, args.lam)))
    for k in "abc":
        print(f"MSE {k}: {np.mean((true[k] - est[k]) ** 2):.5f}")

    fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
    for ax, k in zip(axes, "abc"):
        ax.plot(true[k], label=f"true {k}(t)", lw=2)
        ax.plot(est[k], "--", label=f"estimated {k}(t)", lw=1.5)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("t")
    fig.suptitle(f"Total-variation regularised AR(3) fit (λ = {args.lam})")
    fig.tight_layout()
    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        fig.savefig(args.save, dpi=130)
    else:
        plt.show()


if __name__ == "__main__":
    main()
