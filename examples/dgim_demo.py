"""Demo: the DGIM algorithm -- counting 1s in a sliding window of an endless stream in tiny memory.

Streams bits through a DGIM counter, tracks its estimate against the exact window count, shows the
memory stays logarithmic and the accuracy tightens with more buckets per size. Draws the estimate vs
exact over time and the exponential bucket structure.

    python examples/dgim_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dgim import DGIM, ExactWindow  # noqa: E402


class LCG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def nxt(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return self.s

    def bit(self, pnum, pden):
        return 1 if (self.nxt() >> 8) % pden < pnum else 0


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("DGIM: count the 1s in the last N bits of an infinite stream, in O(log^2 N) memory\n")

    window = 1000
    rng = LCG(2024)
    # a stream whose density drifts, to make the count move
    est_track = []
    exact_track = []
    for r in (2, 4, 8):
        d = DGIM(window, buckets_per_size=r)
        e = ExactWindow(window)
        worst = 0.0
        rng2 = LCG(2024)
        for i in range(20000):
            # density oscillates so the true count sweeps up and down
            pnum = 3 + int(4 * (1 + math.sin(i / 2000)))
            b = rng2.bit(pnum, 10)
            d.push(b)
            e.push(b)
            ex = e.count()
            if ex > 0:
                worst = max(worst, abs(d.count() - ex) / ex)
            if r == 4 and i % 100 == 0:
                est_track.append(d.count())
                exact_track.append(ex)
        print(f"  buckets_per_size={r}: worst relative error {worst:.3f} "
              f"(guaranteed <= {d.error_bound():.2f}), {d.bucket_count()} buckets stored")
    print()

    # memory demonstration
    big = DGIM(1 << 20, buckets_per_size=2)   # ~1 million-bit window
    rng3 = LCG(5)
    for _ in range(2000000):
        big.push(rng3.bit(1, 2))
    print(f"  window of {big.window:,} bits after 2,000,000 pushes:")
    print(f"    stored just {big.bucket_count()} buckets "
          f"(a full window would need {big.window:,} bits)\n")

    print("  Buckets have power-of-two sizes and there are at most a constant number of each, so the")
    print("  window is summarised by O(log^2 N) numbers. The only error comes from halving the oldest")
    print("  bucket, which straddles the window edge -- bounded by 50% with two buckets per size, and")
    print("  ~1/(k-1) with k. The technique behind sliding-window analytics on unbounded streams.")

    _svg(os.path.join(outdir, "dgim.svg"), est_track, exact_track, big)
    print(f"\n  wrote {os.path.join(outdir, 'dgim.svg')}")


def _svg(path, est, exact, dgim_big, width=760, height=440):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="16">'
        f'DGIM estimate vs exact sliding-window count (top); bucket sizes (bottom)</text>',
    ]

    # top: est vs exact over time
    ox, oy = 55, 240
    pw, ph = width - 90, 180
    n = len(exact)
    ymax = max(max(est), max(exact)) if exact else 1
    parts.append(f'<line x1="{ox}" y1="{oy}" x2="{ox+pw}" y2="{oy}" stroke="#30363d" stroke-width="1"/>')
    parts.append(f'<line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy-ph}" stroke="#30363d" stroke-width="1"/>')

    def px(i):
        return ox + i / max(1, n - 1) * pw

    def py(v):
        return oy - v / ymax * ph

    pe = " ".join(f"{px(i):.1f},{py(exact[i]):.1f}" for i in range(n))
    parts.append(f'<polyline points="{pe}" fill="none" stroke="#06d6a0" stroke-width="2"/>')
    pd = " ".join(f"{px(i):.1f},{py(est[i]):.1f}" for i in range(n))
    parts.append(f'<polyline points="{pd}" fill="none" stroke="#ffd43b" stroke-width="1.5" '
                 f'stroke-dasharray="4 3"/>')
    parts.append(f'<text x="{ox+20}" y="{oy-ph+18}" fill="#06d6a0" font-size="12">exact count</text>')
    parts.append(f'<text x="{ox+20}" y="{oy-ph+36}" fill="#ffd43b" font-size="12">DGIM estimate</text>')
    parts.append(f'<text x="{ox+pw/2:.0f}" y="{oy+28}" fill="#8b949e" font-size="12" '
                 f'text-anchor="middle">time (stream position)</text>')

    # bottom: bucket sizes of the big counter as bars (log2 size)
    bx, by = 55, 420
    bw, bh = width - 90, 130
    sizes = [size for _, size in dgim_big.buckets]
    if sizes:
        smax = max(sizes)
        barw = bw / max(1, len(sizes))
        parts.append(f'<text x="{bx}" y="{by-bh-6}" fill="#8b949e" font-size="11">'
                     f'{len(sizes)} buckets, sizes (powers of two) newest -> oldest, '
                     f'window {dgim_big.window:,} bits</text>')
        for i, s in enumerate(sizes):
            h = bh * math.log2(s + 1) / math.log2(smax + 1)
            parts.append(f'<rect x="{bx + i*barw:.1f}" y="{by-h:.1f}" width="{barw-1.5:.1f}" '
                         f'height="{h:.1f}" fill="#4dabf7"/>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
