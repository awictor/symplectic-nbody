"""Demo: Welford's algorithm -- stable running statistics in one pass.

Streams values into an online accumulator, showing the running mean and variance converge to
the truth, then demonstrates the naive formula collapsing on offset data where Welford stays
exact. Draws the running mean/std converging and the naive-vs-Welford variance error growing
with the offset.

    python examples/welford_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from welford import Welford, two_pass_mean_variance, naive_variance  # noqa: E402


def _stream(n, mu, sigma, seed=1):
    """A pseudo-random Gaussian-ish stream via summed uniforms (CLT), seeded."""
    state = seed
    out = []
    for _ in range(n):
        s = 0.0
        for _ in range(12):                 # sum of 12 uniforms ~ N(6, 1)
            state = (1664525 * state + 1013904223) & 0xFFFFFFFF
            s += (state >> 8) / (1 << 24)
        out.append(mu + sigma * (s - 6.0))
    return out


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    data = _stream(5000, mu=50.0, sigma=8.0, seed=1)
    w = Welford().update(data)
    tm, tv = two_pass_mean_variance(data)
    print("Welford: one-pass, online mean and variance, numerically stable\n")
    print(f"  {len(data)} values ~ N(50, 8^2):")
    print(f"    mean     {w.mean:.4f}  (two-pass {tm:.4f})")
    print(f"    variance {w.variance():.4f}  (two-pass {tv:.4f})")
    print(f"    std      {w.std():.4f},  skewness {w.skewness():+.4f},  kurtosis {w.kurtosis():.4f}\n")

    print("  The naive variance E[x^2]-E[x]^2 collapses when the data has a large offset:")
    print(f"  {'offset':>12}{'true var':>10}{'Welford':>12}{'naive':>16}")
    base = [1.0, 2.0, 3.0, 4.0, 5.0]      # variance 2.0
    for off in (0, 1e3, 1e6, 1e9, 1e12):
        shifted = [x + off for x in base]
        wv = Welford().update(shifted).variance()
        nv = naive_variance(shifted)
        print(f"  {off:>12.0e}{2.0:>10.1f}{wv:>12.4f}{nv:>16.4f}")
    print("\n  Welford never forms the giant sum-of-squares, so it stays exact; the naive formula")
    print("  subtracts two nearly-equal huge numbers and loses everything to rounding -- even")
    print("  returning negative variances. Accumulators also merge, so shards combine in parallel.")

    _svg(os.path.join(outdir, "welford.svg"), data, tm, math.sqrt(tv))
    print(f"\n  wrote {os.path.join(outdir, 'welford.svg')}")


def _svg(path, data, true_mean, true_std, w=760, h=400):
    # running mean and std as the stream is consumed
    acc = Welford()
    means, stds = [], []
    for x in data:
        acc.add(x)
        means.append(acc.mean)
        stds.append(acc.std())

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="monospace">',
        f'<rect width="{w}" height="{h}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Welford: running mean and std converge as the stream flows</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'running estimates settle onto the true mean (blue) and std (green) after a few hundred values</text>',
    ]

    n = len(means)
    x0, x1 = 55, w - 30
    y0, y1 = h - 55, 62
    # y range covers both mean and std
    ymin = min(min(means[5:]), min(stds[5:])) * 0.9
    ymax = max(max(means[5:]), max(stds[5:])) * 1.05

    def X(i):
        return x0 + i / (n - 1) * (x1 - x0)

    def Y(v):
        return y0 - (v - ymin) / (ymax - ymin) * (y0 - y1)

    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#8b949e" stroke-width="1.2"/>')
    # true-value reference lines
    parts.append(f'<line x1="{x0}" y1="{Y(true_mean):.1f}" x2="{x1}" y2="{Y(true_mean):.1f}" '
                 f'stroke="#4dabf7" stroke-width="1" stroke-dasharray="4 3" opacity="0.6"/>')
    parts.append(f'<line x1="{x0}" y1="{Y(true_std):.1f}" x2="{x1}" y2="{Y(true_std):.1f}" '
                 f'stroke="#06d6a0" stroke-width="1" stroke-dasharray="4 3" opacity="0.6"/>')
    # running curves (skip the first few wild points)
    mean_pts = " ".join(f"{X(i):.1f},{Y(means[i]):.1f}" for i in range(2, n))
    std_pts = " ".join(f"{X(i):.1f},{Y(stds[i]):.1f}" for i in range(2, n))
    parts.append(f'<polyline points="{mean_pts}" fill="none" stroke="#4dabf7" stroke-width="1.8"/>')
    parts.append(f'<polyline points="{std_pts}" fill="none" stroke="#06d6a0" stroke-width="1.8"/>')
    parts.append(f'<text x="{x1-2:.1f}" y="{Y(true_mean)-5:.1f}" fill="#4dabf7" font-size="9" '
                 f'text-anchor="end">true mean {true_mean:.1f}</text>')
    parts.append(f'<text x="{x1-2:.1f}" y="{Y(true_std)-5:.1f}" fill="#06d6a0" font-size="9" '
                 f'text-anchor="end">true std {true_std:.1f}</text>')
    parts.append(f'<text x="{(x0+x1)/2:.1f}" y="{y0+18:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">values seen -> running estimate</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
