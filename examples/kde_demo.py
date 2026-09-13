"""Demo: kernel density estimation -- a smooth density from samples, no model assumed.

Draws samples from a bimodal mixture, estimates the density with KDE at three bandwidths (showing
under/over-smoothing), compares kernels, and picks a bandwidth by leave-one-out likelihood. Draws the
KDE curves over a rug of the samples and the true density.

    python examples/kde_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kde import KDE, silverman_bandwidth, scott_bandwidth, grid  # noqa: E402


def _lcg(seed):
    state = seed & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 8) / (1 << 24)
    return nxt


def _normal(rng, mu, sigma):
    u1 = max(rng(), 1e-12)
    u2 = rng()
    return mu + sigma * math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Kernel density estimation: the smooth cousin of the histogram\n")

    rng = _lcg(2024)
    # bimodal: 60% around -2, 40% around +2.5
    data = []
    for _ in range(200):
        if rng() < 0.6:
            data.append(_normal(rng, -2, 0.7))
        else:
            data.append(_normal(rng, 2.5, 0.9))

    def true_pdf(x):
        return (0.6 * math.exp(-0.5 * ((x + 2) / 0.7) ** 2) / (0.7 * math.sqrt(2 * math.pi)) +
                0.4 * math.exp(-0.5 * ((x - 2.5) / 0.9) ** 2) / (0.9 * math.sqrt(2 * math.pi)))

    h_silverman = silverman_bandwidth(data)
    h_scott = scott_bandwidth(data)
    print(f"  {len(data)} samples from a bimodal mixture (peaks at -2 and +2.5).")
    print(f"  Silverman bandwidth: {h_silverman:.4f}   Scott bandwidth: {h_scott:.4f}\n")

    print("  Bandwidth choice by leave-one-out log-likelihood:")
    print(f"    {'bandwidth':>10}  {'LOO log-lik':>12}  {'note':>16}")
    best_h, best_ll = None, float("-inf")
    for h in [0.1, 0.2, h_silverman, 0.5, 1.0, 2.0]:
        ll = KDE(data, bandwidth=h).loo_log_likelihood()
        note = "(Silverman)" if abs(h - h_silverman) < 1e-9 else ""
        print(f"    {h:>10.3f}  {ll:>12.1f}  {note:>16}")
        if ll > best_ll:
            best_ll, best_h = ll, h
    print(f"    -> best LOO bandwidth ~ {best_h:.3f}")

    print("\n  Kernel comparison at the left peak (x=-2), Silverman bandwidth:")
    for kname in ["gaussian", "epanechnikov", "triangular", "cosine"]:
        print(f"    {kname:>13}: f_hat(-2) = {KDE(data, kernel=kname).pdf(-2):.4f}  "
              f"(true {true_pdf(-2):.4f})")

    _svg(os.path.join(outdir, "kde.svg"), data, true_pdf, h_silverman)
    print(f"\n  wrote {os.path.join(outdir, 'kde.svg')}")


def _svg(path, data, true_pdf, h_silverman, width=760, height=430):
    lo, hi = -6, 7
    xs = grid(lo, hi, 400)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        '<text x="20" y="26" fill="#e6edf3" font-size="15">'
        'KDE at three bandwidths vs the true bimodal density (gray); samples as a rug</text>',
    ]
    ox, oy, ow, oh = 50, 50, width - 90, height - 120

    curves = [
        (0.15, "#ff6b6b", "h=0.15 (under)"),
        (h_silverman, "#06d6a0", f"h={h_silverman:.2f} (Silverman)"),
        (1.2, "#4dabf7", "h=1.2 (over)"),
    ]
    true_ys = [true_pdf(x) for x in xs]
    all_ys = list(true_ys)
    curve_ys = []
    for h, _, _ in curves:
        ys = KDE(data, bandwidth=h).evaluate(xs)
        curve_ys.append(ys)
        all_ys += ys
    ymax = max(all_ys) * 1.1

    def px(x):
        return ox + ow * (x - lo) / (hi - lo)

    def py(y):
        return oy + oh * (1 - y / ymax)

    # true density
    tp = " ".join(f"{px(xs[i]):.1f},{py(true_ys[i]):.1f}" for i in range(len(xs)))
    parts.append(f'<polyline points="{tp}" fill="none" stroke="#8b949e" stroke-width="2.5" '
                 f'stroke-dasharray="5 3"/>')
    # KDE curves
    for (h, col, label), ys in zip(curves, curve_ys):
        p = " ".join(f"{px(xs[i]):.1f},{py(ys[i]):.1f}" for i in range(len(xs)))
        parts.append(f'<polyline points="{p}" fill="none" stroke="{col}" stroke-width="1.8"/>')
    # rug
    for x in data:
        parts.append(f'<line x1="{px(x):.1f}" y1="{oy+oh}" x2="{px(x):.1f}" y2="{oy+oh+8}" '
                     f'stroke="#ffd43b" stroke-width="0.6" opacity="0.6"/>')
    # legend
    ly = oy + 6
    for h, col, label in curves:
        parts.append(f'<text x="{ox+ow-150}" y="{ly}" fill="{col}" font-size="10">{label}</text>')
        ly += 15
    parts.append(f'<text x="{ox+ow-150}" y="{ly}" fill="#8b949e" font-size="10">true (dashed)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
