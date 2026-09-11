"""Demo: Kahan summation -- adding floats without losing the small ones.

Sums a long stream of 0.1s naively and with compensation, showing the naive error grows with n
while Kahan stays exact, then a catastrophic-cancellation case where only compensated summation
survives. Draws the relative error versus the number of terms for each method.

    python examples/kahan_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kahan import (naive_sum, kahan_sum, neumaier_sum, pairwise_sum,  # noqa: E402
                   relative_error)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Kahan summation: carry the low-order bits lost on each addition\n")
    print("  summing 0.1 repeatedly (exact answer = n/10):")
    print(f"  {'n':>10}{'naive error':>16}{'Kahan error':>14}{'pairwise':>14}")
    for exp in (3, 4, 5, 6, 7):
        n = 10 ** exp
        vals = [0.1] * n
        exact = math.fsum(vals)
        print(f"  {n:>10}{relative_error(naive_sum(vals), exact):>16.2e}"
              f"{relative_error(kahan_sum(vals), exact):>14.2e}"
              f"{relative_error(pairwise_sum(vals), exact):>14.2e}")

    print("\n  Catastrophic cancellation, exact answer = 2.0:")
    ill = [1.0, 1e100, 1.0, -1e100]
    print(f"    input {ill}")
    print(f"    naive     = {naive_sum(ill)}   (both 1.0s vanished)")
    print(f"    Kahan     = {kahan_sum(ill)}")
    print(f"    Neumaier  = {neumaier_sum(ill)}   <- recovers it")
    print(f"    math.fsum = {math.fsum(ill)}")

    print("\n  The naive error grows like n * machine-epsilon; Kahan keeps a correction term so")
    print("  it stays bounded, as if the sum were done in double the precision. It matters for")
    print("  long dot products, running averages, and any accumulation over millions of terms.")

    _svg(os.path.join(outdir, "kahan.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'kahan.svg')}")


def _svg(path, w=760, h=390):
    # relative error vs n for each method (log-log)
    exps = [2, 3, 4, 5, 6, 7]
    ns = [10 ** e for e in exps]
    naive_e, kahan_e, pair_e = [], [], []
    for n in ns:
        vals = [0.1] * n
        exact = math.fsum(vals)
        naive_e.append(max(relative_error(naive_sum(vals), exact), 1e-18))
        kahan_e.append(max(relative_error(kahan_sum(vals), exact), 1e-18))
        pair_e.append(max(relative_error(pairwise_sum(vals), exact), 1e-18))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="monospace">',
        f'<rect width="{w}" height="{h}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Kahan summation: naive error grows, compensated stays flat</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'relative error vs number of 0.1 terms summed (both axes log)</text>',
    ]

    x0, x1 = 60, w - 40
    y0, y1 = h - 55, 62
    # log10 error axis from 1e-18 to 1e-8
    emin, emax = -18.0, -8.0

    def X(n):
        return x0 + (math.log10(n) - exps[0]) / (exps[-1] - exps[0]) * (x1 - x0)

    def Y(err):
        le = max(emin, min(emax, math.log10(err)))
        return y0 - (le - emin) / (emax - emin) * (y0 - y1)

    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#8b949e" stroke-width="1.2"/>')
    for e in range(-18, -7, 2):
        yy = Y(10 ** e)
        parts.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x1}" y2="{yy:.1f}" stroke="#21262d" stroke-width="1"/>')
        parts.append(f'<text x="{x0-6:.1f}" y="{yy+3:.1f}" fill="#8b949e" font-size="8" '
                     f'text-anchor="end">1e{e}</text>')
    for series, col, lab in ((naive_e, "#ff6b6b", "naive"),
                             (pair_e, "#ffd43b", "pairwise"),
                             (kahan_e, "#06d6a0", "Kahan")):
        pts = " ".join(f"{X(n):.1f},{Y(err):.1f}" for n, err in zip(ns, series))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.5"/>')
        for n, err in zip(ns, series):
            parts.append(f'<circle cx="{X(n):.1f}" cy="{Y(err):.1f}" r="2.5" fill="{col}"/>')
    parts.append(f'<text x="{X(ns[-1]):.1f}" y="{Y(naive_e[-1])-6:.1f}" fill="#ff6b6b" font-size="9" '
                 f'text-anchor="end">naive ~ n eps</text>')
    parts.append(f'<text x="{X(ns[-1]):.1f}" y="{Y(kahan_e[-1])+14:.1f}" fill="#06d6a0" font-size="9" '
                 f'text-anchor="end">Kahan (flat)</text>')
    for e in exps:
        parts.append(f'<text x="{X(10**e):.1f}" y="{y0+15:.1f}" fill="#8b949e" font-size="8" '
                     f'text-anchor="middle">1e{e}</text>')
    parts.append(f'<text x="{(x0+x1)/2:.1f}" y="{y0+30:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">number of terms n (log) -> relative error (log)</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
