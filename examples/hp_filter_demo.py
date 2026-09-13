"""Demo: Hodrick-Prescott filter -- separating a trend from a cycle in a noisy series.

Decomposes a synthetic "GDP-like" series (a rising trend + a business-cycle wiggle + noise) into
trend and cycle for several smoothing parameters, showing the lambda trade-off between tracking and
smoothing. Draws the series with its extracted trend and the cycle below.

    python examples/hp_filter_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hp_filter import hp_filter, second_difference_norm  # noqa: E402


def _lcg(seed):
    state = seed & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 8) / (1 << 24)
    return nxt


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Hodrick-Prescott filter: split a time series into a smooth trend and a cycle\n")

    rng = _lcg(2024)
    n = 80
    # rising trend + slow cycle + noise
    y = []
    for t in range(n):
        trend = 100 + 0.8 * t + 0.01 * t * t
        cycle = 4 * math.sin(2 * math.pi * t / 20)
        noise = (rng() - 0.5) * 3
        y.append(trend + cycle + noise)

    print(f"  Synthetic series ({n} points): quadratic trend + 20-period cycle + noise.\n")
    print(f"  {'lambda':>10}  {'trend roughness':>16}  {'cycle std':>10}")
    for lam in [10, 100, 1600, 10000, 100000]:
        tau, cyc = hp_filter(y, lam)
        rough = second_difference_norm(tau)
        cstd = (sum(c * c for c in cyc) / n) ** 0.5
        note = "  <- classic quarterly lambda" if lam == 1600 else ""
        print(f"  {lam:>10}  {rough:>16.4f}  {cstd:>10.3f}{note}")

    print("\n  Small lambda tracks every wiggle (trend ~ data, tiny cycle); large lambda gives an")
    print("  almost-straight trend and pushes the cycle up. Lambda=1600 is the textbook quarterly")
    print("  choice. The trend minimizes fidelity + lambda * (second-difference)^2 exactly.")

    _svg(os.path.join(outdir, "hp_filter.svg"), y, *hp_filter(y, 1600))
    print(f"\n  wrote {os.path.join(outdir, 'hp_filter.svg')}")


def _svg(path, y, trend, cycle, width=760, height=430):
    n = len(y)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        '<text x="20" y="24" fill="#e6edf3" font-size="15">'
        'Top: series (gray) with HP trend (green, lambda=1600). Bottom: extracted cycle</text>',
    ]

    # top panel: series + trend
    tx0, ty0, tw, th = 40, 45, width - 80, 200
    ylo = min(min(y), min(trend))
    yhi = max(max(y), max(trend))

    def sx(i):
        return tx0 + tw * i / (n - 1)

    def ty(v):
        return ty0 + th * (1 - (v - ylo) / (yhi - ylo))

    for i in range(n):
        parts.append(f'<circle cx="{sx(i):.1f}" cy="{ty(y[i]):.1f}" r="1.8" fill="#8b949e" '
                     f'opacity="0.7"/>')
    tp = " ".join(f"{sx(i):.1f},{ty(trend[i]):.1f}" for i in range(n))
    parts.append(f'<polyline points="{tp}" fill="none" stroke="#06d6a0" stroke-width="2"/>')

    # bottom panel: cycle
    bx0, by0, bw, bh = 40, 290, width - 80, 110
    clo, chi = min(cycle), max(cycle)
    span = max(abs(clo), abs(chi)) or 1

    def cy(v):
        return by0 + bh / 2 * (1 - v / span)

    parts.append(f'<line x1="{bx0}" y1="{cy(0):.1f}" x2="{bx0+bw}" y2="{cy(0):.1f}" '
                 f'stroke="#30363d" stroke-width="1"/>')
    cp = " ".join(f"{bx0 + bw*i/(n-1):.1f},{cy(cycle[i]):.1f}" for i in range(n))
    parts.append(f'<polyline points="{cp}" fill="none" stroke="#ff922b" stroke-width="1.5"/>')
    parts.append(f'<text x="{bx0}" y="{by0-6}" fill="#8b949e" font-size="10">'
                 f'cycle = data - trend (the business-cycle component)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
