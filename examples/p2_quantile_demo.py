"""Demo: the P-square algorithm estimating latency percentiles from a stream, in O(1) memory.

Streams a heavy-tailed latency distribution through P-square estimators for p50/p90/p95/p99, compares
to the exact quantiles, and shows the memory saving. Draws the estimate converging to the true value
as the stream grows.

    python examples/p2_quantile_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from p2_quantile import P2Quantile, P2Histogram  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("P-square: streaming quantiles (latency percentiles) in constant memory\n")

    state = 12345

    def rng():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 16) / 65536.0

    # a heavy-tailed latency stream: mostly fast, occasional slow requests (log-normal-ish)
    n = 200000
    latencies = []
    for _ in range(n):
        u = max(rng(), 1e-12)
        # base ~20ms, heavy tail
        latencies.append(20 * math.exp(1.1 * (-math.log(u)) ** 0.5))

    hist = P2Histogram(quantiles=(0.5, 0.9, 0.95, 0.99))
    for x in latencies:
        hist.add(x)

    exact = {}
    s = sorted(latencies)
    for p in (0.5, 0.9, 0.95, 0.99):
        exact[p] = s[int(p * len(s))]

    print(f"  streamed {n} latency samples (heavy-tailed, in ms):")
    print(f"    {'percentile':>10}  {'P-square':>10}  {'exact':>10}  {'error':>8}")
    for p in (0.5, 0.9, 0.95, 0.99):
        est = hist.quantile(p)
        print(f"    {'p'+str(int(p*100)):>10}  {est:>10.2f}  {exact[p]:>10.2f}  "
              f"{abs(est-exact[p])/exact[p]*100:>7.2f}%")

    print(f"\n  Memory: P-square keeps 5 markers per quantile = {4*5} floats total,")
    print(f"  versus {n} samples ({n*8//1024} KB) that exact computation would need to store and sort.")

    # convergence of the p95 estimate as the stream grows
    print("\n  The p95 estimate converges to the true value as more data arrives:")
    est = P2Quantile(0.95)
    checkpoints = [100, 1000, 10000, 50000, 200000]
    conv = []
    ci = 0
    for i, x in enumerate(latencies, 1):
        est.add(x)
        if ci < len(checkpoints) and i == checkpoints[ci]:
            true_so_far = sorted(latencies[:i])[int(0.95 * i)]
            conv.append((i, est.quantile(), true_so_far))
            print(f"    after {i:6d} samples: p95 estimate {est.quantile():7.2f}  "
                  f"(true so far {true_so_far:7.2f})")
            ci += 1

    print("\n  Five markers track the min, max, the target quantile, and two midpoints; each new")
    print("  sample nudges the markers toward their desired positions using parabolic interpolation.")
    print("  No samples are stored -- ideal for p99 latency monitoring on a firehose of requests.")

    _svg(os.path.join(outdir, "p2_quantile.svg"), conv, exact)
    print(f"\n  wrote {os.path.join(outdir, 'p2_quantile.svg')}")


def _svg(path, conv, exact, width=760, height=430):
    m_left, m_bot, m_top, m_right = 70, 60, 80, 40
    pw = width - m_left - m_right
    ph = height - m_top - m_bot

    xs = [math.log10(c[0]) for c in conv]
    xmin, xmax = min(xs), max(xs)
    true_p95 = exact[0.95]
    all_y = [c[1] for c in conv] + [c[2] for c in conv] + [true_p95]
    ymin, ymax = min(all_y) * 0.9, max(all_y) * 1.1

    def px(lx):
        return m_left + (lx - xmin) / (xmax - xmin) * pw

    def py(v):
        return m_top + ph - (v - ymin) / (ymax - ymin) * ph

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">'
        f'P-square: the p95 estimate converging with stream size</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'blue = P-square estimate, yellow dashed = the true final p95 (constant memory throughout)</text>',
    ]

    parts.append(f'<line x1="{m_left}" y1="{m_top+ph}" x2="{m_left+pw}" y2="{m_top+ph}" '
                 f'stroke="#484f58" stroke-width="1.5"/>')
    parts.append(f'<line x1="{m_left}" y1="{m_top}" x2="{m_left}" y2="{m_top+ph}" '
                 f'stroke="#484f58" stroke-width="1.5"/>')

    # true p95 reference line
    parts.append(f'<line x1="{m_left}" y1="{py(true_p95):.1f}" x2="{m_left+pw}" y2="{py(true_p95):.1f}" '
                 f'stroke="#ffd43b" stroke-width="1.8" stroke-dasharray="6,4"/>')
    parts.append(f'<text x="{m_left+pw-4}" y="{py(true_p95)-6:.1f}" fill="#ffd43b" font-size="11" '
                 f'text-anchor="end">true p95 = {true_p95:.1f} ms</text>')

    est_pts = " ".join(f"{px(lx):.1f},{py(c[1]):.1f}" for lx, c in zip(xs, conv))
    parts.append(f'<polyline points="{est_pts}" fill="none" stroke="#4dabf7" stroke-width="2.2"/>')
    for lx, c in zip(xs, conv):
        parts.append(f'<circle cx="{px(lx):.1f}" cy="{py(c[1]):.1f}" r="4" fill="#4dabf7"/>')
        parts.append(f'<text x="{px(lx):.0f}" y="{m_top+ph+18:.0f}" fill="#8b949e" font-size="10" '
                     f'text-anchor="middle">{c[0]}</text>')

    parts.append(f'<text x="{m_left+pw/2:.0f}" y="{height-12}" fill="#8b949e" font-size="12" '
                 f'text-anchor="middle">samples seen (log scale)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
