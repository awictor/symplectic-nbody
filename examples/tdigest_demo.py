"""Demo: the t-digest -- accurate streaming quantiles over the whole distribution.

Streams a large skewed dataset through a small t-digest, compares its p50/p90/p99/p999 to the exact
sorted answers, shows the memory savings and the tail sharpness, demonstrates distributed merging, and
draws the estimated vs exact quantile curve.

    python examples/tdigest_demo.py [output_dir]
"""

import bisect
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tdigest import TDigest, merge_digests  # noqa: E402


class LCG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def u(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return (self.s >> 8) / (1 << 24)

    def expo(self, lam=1.0):
        return -math.log(1 - self.u()) / lam


def exact_q(sd, q):
    idx = q * (len(sd) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sd) - 1)
    return sd[lo] * (1 - (idx - lo)) + sd[hi] * (idx - lo)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("t-digest: p50/p90/p99/p999 from one pass, no storing the stream\n")

    rng = LCG(2024)
    n = 500000
    # simulated latencies: mostly fast, long exponential tail (ms)
    data = [5.0 + rng.expo(0.05) for _ in range(n)]
    td = TDigest(delta=100)
    td.add_all(data)
    sd = sorted(data)

    print(f"  streamed {n:,} latency samples through a digest of {td.centroid_count()} centroids")
    print(f"    memory: {td.centroid_count()} centroids vs {n:,} raw values "
          f"(~{n // td.centroid_count():,}x smaller)\n")

    print("  quantile      t-digest      exact      rank error")
    for q in (0.5, 0.9, 0.99, 0.999, 0.9999):
        est = td.quantile(q)
        ex = exact_q(sd, q)
        rank = bisect.bisect_left(sd, est) / len(sd)
        print(f"    p{q*100:<7.2f}  {est:10.3f}  {ex:10.3f}    {abs(rank-q):.5f}")
    print()

    # distributed merge
    shards = [TDigest(100) for _ in range(8)]
    for i, v in enumerate(data):
        shards[i % 8].add(v)
    merged = merge_digests(shards, 100)
    print("  distributed roll-up: 8 per-shard digests merged into one")
    print(f"    merged p99 = {merged.quantile(0.99):.3f}  vs single-digest p99 = {td.quantile(0.99):.3f}")
    print(f"    vs exact p99 = {exact_q(sd, 0.99):.3f}\n")

    print("  The scale function k(q) = (delta/2pi) arcsin(2q-1) is compressed at the tails, so")
    print("  centroids there hold almost no weight (fine resolution) while a single centroid near the")
    print("  median absorbs a big slice (coarse, but who cares). Merges are associative -> distributed.")

    _svg(os.path.join(outdir, "tdigest.svg"), td, sd)
    print(f"\n  wrote {os.path.join(outdir, 'tdigest.svg')}")


def _svg(path, td, sd, width=760, height=400):
    ox, oy = 60, 330
    pw, ph = width - 100, 270

    qs = [i / 100 for i in range(101)]
    est = [td.quantile(q) for q in qs]
    exact = [exact_q(sd, q) for q in qs]
    ymin = min(min(est), min(exact))
    ymax = max(max(est), max(exact))
    span = ymax - ymin if ymax > ymin else 1.0

    def px(q):
        return ox + q * pw

    def py(v):
        return oy - (v - ymin) / span * ph

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="17">'
        f't-digest quantile function vs exact -- a heavy-tailed latency stream</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'the estimate (dashed) tracks the exact curve (solid), tightest at the steep upper tail</text>',
    ]
    parts.append(f'<line x1="{ox}" y1="{oy}" x2="{ox+pw}" y2="{oy}" stroke="#30363d" stroke-width="1"/>')
    parts.append(f'<line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy-ph}" stroke="#30363d" stroke-width="1"/>')

    ptexact = " ".join(f"{px(q):.1f},{py(v):.1f}" for q, v in zip(qs, exact))
    parts.append(f'<polyline points="{ptexact}" fill="none" stroke="#06d6a0" stroke-width="2"/>')
    ptest = " ".join(f"{px(q):.1f},{py(v):.1f}" for q, v in zip(qs, est))
    parts.append(f'<polyline points="{ptest}" fill="none" stroke="#ffd43b" stroke-width="1.6" '
                 f'stroke-dasharray="5 3"/>')

    # mark the tail quantiles
    for q in (0.5, 0.9, 0.99):
        parts.append(f'<line x1="{px(q):.1f}" y1="{oy}" x2="{px(q):.1f}" y2="{oy-ph}" '
                     f'stroke="#30363d" stroke-width="0.7" stroke-dasharray="2 4"/>')
        parts.append(f'<text x="{px(q):.0f}" y="{oy+16}" fill="#8b949e" font-size="10" '
                     f'text-anchor="middle">p{int(q*100)}</text>')

    parts.append(f'<text x="{ox+40}" y="{oy-ph+18}" fill="#06d6a0" font-size="12">exact</text>')
    parts.append(f'<text x="{ox+40}" y="{oy-ph+36}" fill="#ffd43b" font-size="12">t-digest</text>')
    parts.append(f'<text x="{ox+pw/2:.0f}" y="{oy+34}" fill="#8b949e" font-size="12" '
                 f'text-anchor="middle">quantile q</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
