"""
可视化 BCA 采样 YAML 的覆盖分布。

用法:
  python plot_bca_samples.py bca_workloads.yaml
  python plot_bca_samples.py bca_workloads.yaml -o bca_coverage.png
"""

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml


def load_samples(yaml_path):
    """从 YAML 加载 test_cases，返回 per-request 空间 (B, c, a)。"""
    with open(yaml_path) as f:
        doc = yaml.safe_load(f)

    cases = doc.get("test_cases", [])
    if not cases:
        print("No test_cases found in YAML.", file=sys.stderr)
        sys.exit(1)

    records = []
    for tc in cases:
        B = tc["batch_size"]
        C = tc["compute_tokens"]
        A = tc["access_tokens"]
        c = C // B
        a = A // B if B > 0 else 0
        rec = {"B": B, "C": C, "A": A, "c": c, "a": a}
        if tc.get("layer") == "boundary":
            rec["is_boundary"] = True
        records.append(rec)

    meta = {
        "token_budget": doc.get("token_budget"),
        "max_num_seqs": doc.get("max_num_seqs"),
        "max_model_len": doc.get("max_model_len"),
    }
    return records, meta


def classify_layer(rec):
    """根据 (c, a) 特征推断所属层（互斥三分 + 边界）。

    边界点由 is_boundary 标志判断，其余按 (c, a) 分类。
    """
    if rec.get("is_boundary"):
        return "boundary"
    c, a = rec["c"], rec["a"]
    if c == 1 and a >= 1:
        return "decode"
    if a == 0:
        return "prefill"
    return "chunked"


LAYER_COLORS = {
    "boundary": "#d62728",
    "decode": "#1f77b4",
    "prefill": "#ff7f0e",
    "chunked": "#2ca02c",
}
LAYER_ORDER = ["boundary", "decode", "prefill", "chunked"]


def plot_coverage(records, meta, output_path):
    """生成 2x2 覆盖分布图。"""
    # 分层
    by_layer = {k: [] for k in LAYER_ORDER}
    for r in records:
        layer = classify_layer(r)
        by_layer[layer].append(r)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        f"BCA Sample Coverage  (n={len(records)}, "
        f"budget={meta.get('token_budget')}, "
        f"max_seqs={meta.get('max_num_seqs')}, "
        f"max_len={meta.get('max_model_len')})",
        fontsize=13, fontweight="bold",
    )

    # 绘制辅助: boundary 用大菱形，其余用小圆点
    def _layer_style(layer):
        if layer == "boundary":
            return dict(s=48, marker="D", alpha=0.9, edgecolors="k",
                        linewidths=0.5, zorder=5)
        return dict(s=12, marker="o", alpha=0.6, edgecolors="none", zorder=3)

    # ---------- ax0: B vs c (per-request compute) ----------
    ax = axes[0, 0]
    for layer in LAYER_ORDER:
        pts = by_layer[layer]
        if not pts:
            continue
        xs = [p["B"] for p in pts]
        ys = [max(p["c"], 0.8) for p in pts]
        ax.scatter(xs, ys, label=f"{layer} ({len(pts)})",
                   color=LAYER_COLORS[layer], **_layer_style(layer))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("B  (batch_size)")
    ax.set_ylabel("c  (compute / request)")
    ax.set_title("B vs c")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3, which="both")

    # ---------- ax1: B vs a (per-request access) ----------
    ax = axes[0, 1]
    for layer in LAYER_ORDER:
        pts = by_layer[layer]
        if not pts:
            continue
        style = _layer_style(layer)
        has_a = [p for p in pts if p["a"] > 0]
        no_a = [p for p in pts if p["a"] == 0]
        if has_a:
            ax.scatter([p["B"] for p in has_a],
                       [p["a"] for p in has_a],
                       label=f"{layer} ({len(pts)})",
                       color=LAYER_COLORS[layer], **style)
        if no_a:
            s_x = style.copy()
            s_x.update(marker="x", linewidths=0.8)
            if "edgecolors" in s_x:
                del s_x["edgecolors"]
            ax.scatter([p["B"] for p in no_a],
                       [0.8] * len(no_a),
                       color=LAYER_COLORS[layer], **s_x)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("B  (batch_size)")
    ax.set_ylabel("a  (access / request)    [x = a=0]")
    ax.set_title("B vs a")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3, which="both")

    # ---------- ax2: c vs a ----------
    ax = axes[1, 0]
    for layer in LAYER_ORDER:
        pts = by_layer[layer]
        if not pts:
            continue
        style = _layer_style(layer)
        has_a = [p for p in pts if p["a"] > 0]
        no_a = [p for p in pts if p["a"] == 0]
        if has_a:
            ax.scatter([max(p["c"], 0.8) for p in has_a],
                       [p["a"] for p in has_a],
                       label=f"{layer} ({len(pts)})",
                       color=LAYER_COLORS[layer], **style)
        if no_a:
            s_x = style.copy()
            s_x.update(marker="x", linewidths=0.8)
            if "edgecolors" in s_x:
                del s_x["edgecolors"]
            ax.scatter([max(p["c"], 0.8) for p in no_a],
                       [0.8] * len(no_a),
                       color=LAYER_COLORS[layer], **s_x)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("c  (compute / request)    [x = a=0]")
    ax.set_ylabel("a  (access / request)")
    ax.set_title("c vs a  (per-request space)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3, which="both")

    # ---------- ax3: marginal histograms ----------
    ax = axes[1, 1]
    Bs = np.array([r["B"] for r in records], dtype=float)
    cs = np.array([max(r["c"], 1) for r in records], dtype=float)
    # a=0 的 prefill 样本不参与 a 直方图，避免误导
    a_s = np.array([r["a"] for r in records if r["a"] > 0], dtype=float)

    bins_B = np.logspace(0, np.log2(max(Bs)), 20, base=2)
    bins_c = np.logspace(0, np.log2(max(cs)), 20, base=2)
    bins_a = np.logspace(0, np.log2(max(a_s)), 20, base=2) if len(a_s) > 0 else [1]

    ax.hist(Bs, bins=bins_B, alpha=0.5, label="B", color="#1f77b4")
    ax.hist(cs, bins=bins_c, alpha=0.5, label="c", color="#ff7f0e")
    ax.hist(a_s, bins=bins_a, alpha=0.5, label=f"a (>0, n={len(a_s)})", color="#2ca02c")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("value")
    ax.set_ylabel("count")
    ax.set_title("Marginal Distributions (log-x)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize BCA sample coverage from a YAML file",
    )
    parser.add_argument("yaml_file", help="Path to the BCA workloads YAML")
    parser.add_argument("-o", "--output", default=None,
                        help="Output image path (default: <yaml_stem>_coverage.png)")
    args = parser.parse_args()

    if args.output is None:
        stem = args.yaml_file.rsplit(".", 1)[0]
        args.output = f"{stem}_coverage.png"

    records, meta = load_samples(args.yaml_file)
    plot_coverage(records, meta, args.output)


if __name__ == "__main__":
    main()
