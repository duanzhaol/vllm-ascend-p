"""Visualize BCA profiling results from results.csv.

Generates a 2x3 subplot figure showing step_time relationships with
batch size (B), compute tokens (C), and access tokens (A).

Usage:
    python visualize_results.py --input results.csv -o results_analysis.png
"""

import argparse
import csv

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path):
    """Read results.csv into numpy arrays (no pandas dependency)."""
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    n = len(rows)
    B = np.empty(n)
    C = np.empty(n)
    A = np.empty(n)
    avg = np.empty(n)
    std = np.empty(n)

    for i, row in enumerate(rows):
        B[i] = float(row["batch_size"])
        C[i] = float(row["compute_tokens"])
        A[i] = float(row["access_tokens"])
        avg[i] = float(row["avg_step_time_ms"])
        std[i] = float(row["std_step_time_ms"])

    return B, C, A, avg, std


def main():
    parser = argparse.ArgumentParser(
        description="Visualize BCA profiling results")
    parser.add_argument("--input", required=True, help="Path to results.csv")
    parser.add_argument("-o", "--output", default="results_analysis.png",
                        help="Output PNG path")
    args = parser.parse_args()

    B, C, A, avg, std = read_csv(args.input)
    n = len(B)

    # Derived quantities
    c = C / B          # per-request compute tokens
    a = A / B + 1      # per-request access tokens (+1 to avoid log(0))
    cv = std / avg      # coefficient of variation

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        f"BCA Decode Step Profiling Results  (n={n}, PP=4, TP=1)",
        fontsize=14, fontweight="bold", y=0.98,
    )

    # ── Plot 1 (top-left): step_time vs C ──────────────────────────
    ax = axes[0, 0]
    # Color by B (log)
    norm_b = mcolors.LogNorm(vmin=max(B.min(), 1), vmax=B.max())
    sc1 = ax.scatter(np.maximum(C, 1), avg, c=B, cmap="viridis",
                     norm=norm_b, s=24, alpha=0.8, edgecolors="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("C (total compute tokens)")
    ax.set_ylabel("avg_step_time_ms")
    ax.set_title("step_time vs C")
    cb1 = fig.colorbar(sc1, ax=ax, pad=0.02)
    cb1.set_label("B (batch size)")
    ax.grid(True, which="both", ls=":", alpha=0.4)

    # ── Plot 2 (top-center): step_time vs B ────────────────────────
    ax = axes[0, 1]
    norm_c = mcolors.LogNorm(vmin=max(c.min(), 1), vmax=max(c.max(), 2))
    sc2 = ax.scatter(B, avg, c=c, cmap="plasma", norm=norm_c,
                     s=24, alpha=0.8, edgecolors="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("B (batch size)")
    ax.set_ylabel("avg_step_time_ms")
    ax.set_title("step_time vs B")
    cb2 = fig.colorbar(sc2, ax=ax, pad=0.02)
    cb2.set_label("c = C/B (per-request compute)")
    ax.grid(True, which="both", ls=":", alpha=0.4)

    # ── Plot 3 (top-right): step_time vs a ─────────────────────────
    ax = axes[0, 2]
    sc3 = ax.scatter(a, avg, c=c, cmap="plasma", norm=norm_c,
                     s=24, alpha=0.8, edgecolors="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("a = A/B + 1 (per-request access tokens)")
    ax.set_ylabel("avg_step_time_ms")
    ax.set_title("step_time vs a (per-request access)")
    cb3 = fig.colorbar(sc3, ax=ax, pad=0.02)
    cb3.set_label("c = C/B (per-request compute)")
    ax.grid(True, which="both", ls=":", alpha=0.4)

    # ── Plot 4 (bottom-left): performance surface c vs a ───────────
    ax = axes[1, 0]
    norm_t = mcolors.LogNorm(vmin=avg.min(), vmax=avg.max())
    sizes = 10 + 40 * np.log1p(B) / np.log1p(B.max())
    sc4 = ax.scatter(np.maximum(c, 1), a, c=avg, cmap="hot_r", norm=norm_t,
                     s=sizes, alpha=0.8, edgecolors="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("c = C/B (per-request compute)")
    ax.set_ylabel("a = A/B + 1 (per-request access)")
    ax.set_title("Performance surface (c vs a)")
    cb4 = fig.colorbar(sc4, ax=ax, pad=0.02)
    cb4.set_label("avg_step_time_ms")
    ax.grid(True, which="both", ls=":", alpha=0.4)

    # ── Plot 5 (bottom-center): CV distribution ────────────────────
    ax = axes[1, 1]
    ax.hist(cv, bins=40, color="steelblue", edgecolor="white", linewidth=0.5)
    n_high = int(np.sum(cv > 0.1))
    ax.axvline(0.1, color="red", ls="--", lw=1.2, label=f"CV=0.1 ({n_high} samples above)")
    ax.set_xlabel("Coefficient of Variation (std / avg)")
    ax.set_ylabel("Count")
    ax.set_title("Measurement Stability — CV Distribution")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", ls=":", alpha=0.4)

    # ── Plot 6 (bottom-right): actual vs predicted (poly log model) ─
    ax = axes[1, 2]
    # Features: log(B+1), log(c+1), log(a+1) with quadratic & interaction
    log_y = np.log(avg)
    log_B = np.log(B + 1)
    log_c = np.log(c + 1)  # c = C/B (per-request compute)
    log_a = np.log(a)       # a = A/B + 1 (already shifted)
    # Design matrix: 1, B, c, a, Bc, Ba, ca, B², c², a²
    X = np.column_stack([
        np.ones(n), log_B, log_c, log_a,
        log_B * log_c, log_B * log_a, log_c * log_a,
        log_B**2, log_c**2, log_a**2,
    ])
    # Least-squares solve
    coeffs, residuals, rank, sv = np.linalg.lstsq(X, log_y, rcond=None)
    pred_log = X @ coeffs
    pred = np.exp(pred_log)

    # R²
    ss_res = np.sum((log_y - pred_log) ** 2)
    ss_tot = np.sum((log_y - log_y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot

    ax.scatter(pred, avg, s=20, alpha=0.7, edgecolors="none", color="teal")
    lims = [min(pred.min(), avg.min()) * 0.8,
            max(pred.max(), avg.max()) * 1.2]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.6, label="y = x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Predicted step_time (ms)")
    ax.set_ylabel("Actual step_time (ms)")
    ax.set_title("Actual vs Predicted (poly-log model)")
    ax.text(0.05, 0.92,
            f"$R^2 = {r2:.3f}$\n"
            f"features: $\\log B,\\;\\log c,\\;\\log a$\n"
            f"+ quadratic & interactions (10 terms)",
            transform=ax.transAxes, fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, which="both", ls=":", alpha=0.4)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved → {args.output}  ({n} data points, R²={r2:.3f})")


if __name__ == "__main__":
    main()
