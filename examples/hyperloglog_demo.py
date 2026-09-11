"""Demo: HyperLogLog -- counting distinct items in kilobytes.

Estimates the number of distinct items across a range of true cardinalities in fixed tiny
memory, shows the error staying near 1.04/sqrt(m), and demonstrates merging two sketches into
their union. Draws the estimate tracking the truth and the relative error hugging the standard
error band.

    python examples/hyperloglog_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hyperloglog import HyperLogLog, relative_error  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    p = 12
    hll = HyperLogLog(p)
    print(f"HyperLogLog: distinct-count with {hll.m} registers = {hll.m} bytes, "
          f"standard error {hll.standard_error():.2%}\n")
    print(f"  {'true distinct':>14}{'estimate':>12}{'rel error':>12}")
    for n in (100, 1000, 10000, 100000, 500000):
        h = HyperLogLog(p)
        for i in range(n):
            h.add(f"item-{i}")
        est = h.estimate()
        print(f"  {n:>14}{est:>12.0f}{relative_error(est, n):>+12.4f}")

    print(f"\n  Memory stays fixed at ~{hll.m} bytes whether you count 100 or 500,000 items --")
    print("  exact counting would need a set holding all of them. A billion distinct items")
    print("  fit in ~1.5 KB. Sketches merge by register-wise max, so counts are distributed.\n")

    a, b = HyperLogLog(p), HyperLogLog(p)
    for i in range(60000):
        a.add(f"u-{i}")
    for i in range(40000, 100000):  # overlap 20000 -> union 100000
        b.add(f"u-{i}")
    u = a.merge(b)
    print(f"  merge: |A|~{a.estimate():.0f} (true 60000), |B|~{b.estimate():.0f} (true 60000)")
    print(f"         |A union B|~{u.estimate():.0f} (true 100000) -- counted without a shared list.")

    _svg(os.path.join(outdir, "hyperloglog.svg"), p)
    print(f"\n  wrote {os.path.join(outdir, 'hyperloglog.svg')}")


def _svg(path, p, w=760, h=390):
    hll = HyperLogLog(p)
    se = hll.standard_error()

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="monospace">',
        f'<rect width="{w}" height="{h}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'HyperLogLog: distinct count in fixed tiny memory</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'estimate tracks the truth over 4 orders of magnitude (left); '
        f'relative error stays near 1.04/sqrt(m) (right)</text>',
    ]

    # sample true cardinalities (log-spaced) and estimate each
    ns = [100, 300, 1000, 3000, 10000, 30000, 100000, 300000]
    ests = []
    for n in ns:
        hx = HyperLogLog(p)
        for i in range(n):
            hx.add(f"z-{i}")
        ests.append(hx.estimate())

    # left: estimate vs truth on a log-log plot (should hug the diagonal)
    lx0, lx1 = 60, w // 2 - 20
    ly0, ly1 = h - 55, 62
    lo, hi = math.log10(ns[0]), math.log10(ns[-1])

    def LX(v):
        return lx0 + (math.log10(v) - lo) / (hi - lo) * (lx1 - lx0)

    def LY(v):
        return ly0 - (math.log10(v) - lo) / (hi - lo) * (ly0 - ly1)

    parts.append(f'<line x1="{lx0}" y1="{ly0}" x2="{lx1}" y2="{ly0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{lx0}" y1="{ly0}" x2="{lx0}" y2="{ly1}" stroke="#8b949e" stroke-width="1.2"/>')
    # perfect-estimate diagonal
    parts.append(f'<line x1="{LX(ns[0]):.1f}" y1="{LY(ns[0]):.1f}" x2="{LX(ns[-1]):.1f}" '
                 f'y2="{LY(ns[-1]):.1f}" stroke="#8b949e" stroke-width="1" stroke-dasharray="4 3"/>')
    parts.append(f'<text x="{LX(ns[-1]):.1f}" y="{LY(ns[-1])-6:.1f}" fill="#8b949e" font-size="9" '
                 f'text-anchor="end">exact</text>')
    dots = " ".join(f"{LX(n):.1f},{LY(e):.1f}" for n, e in zip(ns, ests))
    parts.append(f'<polyline points="{dots}" fill="none" stroke="#4dabf7" stroke-width="2"/>')
    for n, e in zip(ns, ests):
        parts.append(f'<circle cx="{LX(n):.1f}" cy="{LY(e):.1f}" r="3" fill="#4dabf7"/>')
    for n in (100, 10000, 100000):
        parts.append(f'<text x="{LX(n):.1f}" y="{ly0+15:.1f}" fill="#8b949e" font-size="8" '
                     f'text-anchor="middle">{n:,}</text>')
    parts.append(f'<text x="{(lx0+lx1)/2:.1f}" y="{ly0+30:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">true distinct (log)</text>')
    parts.append(f'<text x="{lx0+4:.1f}" y="{ly1-2:.1f}" fill="#8b949e" font-size="9">estimate (log)</text>')

    # right: relative error with +-SE and +-2SE bands
    rx0, rx1 = w // 2 + 45, w - 30
    ry0, ry1 = h - 55, 62
    ymax = 3 * se

    def RX(i):
        return rx0 + i / (len(ns) - 1) * (rx1 - rx0)

    def RY(err):
        return (ry0 + ry1) / 2 - err / ymax * ((ry0 - ry1) / 2)

    mid = (ry0 + ry1) / 2
    # bands
    for band, col in ((se, "#264d3a"), (2 * se, "#1c2b22")):
        parts.append(f'<rect x="{rx0}" y="{RY(band):.1f}" width="{rx1-rx0:.1f}" '
                     f'height="{RY(-band)-RY(band):.1f}" fill="{col}" opacity="0.6"/>')
    parts.append(f'<line x1="{rx0}" y1="{mid:.1f}" x2="{rx1}" y2="{mid:.1f}" stroke="#8b949e" stroke-width="1"/>')
    parts.append(f'<line x1="{rx0}" y1="{ry1}" x2="{rx0}" y2="{ry0}" stroke="#8b949e" stroke-width="1.2"/>')
    errs = [relative_error(e, n) for n, e in zip(ns, ests)]
    ep = " ".join(f"{RX(i):.1f},{RY(er):.1f}" for i, er in enumerate(errs))
    parts.append(f'<polyline points="{ep}" fill="none" stroke="#ffd43b" stroke-width="2"/>')
    for i, er in enumerate(errs):
        parts.append(f'<circle cx="{RX(i):.1f}" cy="{RY(er):.1f}" r="2.5" fill="#ffd43b"/>')
    parts.append(f'<text x="{rx1-2:.1f}" y="{RY(se)-3:.1f}" fill="#06d6a0" font-size="9" '
                 f'text-anchor="end">+1 SE ({se:.1%})</text>')
    parts.append(f'<text x="{rx1-2:.1f}" y="{RY(-se)+10:.1f}" fill="#06d6a0" font-size="9" '
                 f'text-anchor="end">-1 SE</text>')
    parts.append(f'<text x="{(rx0+rx1)/2:.1f}" y="{ry0+18:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">cardinality sample -> relative error</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
