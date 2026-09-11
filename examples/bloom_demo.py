"""Demo: Bloom filter -- probabilistic membership in a fraction of the space.

Builds a filter, confirms it never gives a false negative, and shows the observed
false-positive rate tracking theory as the filter fills. Draws the false-positive rate rising
with load and how the optimal number of hash functions minimizes it.

    python examples/bloom_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bloom import (BloomFilter, optimal_num_bits, optimal_num_hashes,  # noqa: E402
                   false_positive_rate)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    n, target = 1000, 0.01
    m = optimal_num_bits(n, target)
    k = optimal_num_hashes(m, n)
    print("Bloom filter: 'possibly present' or 'definitely absent', never a false negative\n")
    print(f"  {n} items at {target:.0%} target: {m} bits ({m / n:.1f} bits/item), {k} hash functions")
    print(f"  vs storing the items themselves: a fraction of the memory, item-size-independent\n")

    bf = BloomFilter(m, k)
    items = [f"user:{i}@example.com" for i in range(n)]
    for x in items:
        bf.add(x)
    fn = sum(1 for x in items if x not in bf)
    never = [f"absent:{i}" for i in range(20000)]
    fp = sum(1 for x in never if x in bf) / len(never)
    print(f"  after inserting {n}: false negatives = {fn} (guaranteed 0)")
    print(f"  observed false-positive rate = {fp:.4f},  theory = {false_positive_rate(m, n, k):.4f}")
    print(f"  fill ratio = {bf.fill_ratio():.3f} (optimal load sets ~half the bits)\n")

    print("  False-positive rate as the filter fills (fixed m, k):")
    print(f"  {'items':>8}{'theory':>10}{'observed':>10}")
    for load in (250, 500, 1000, 2000, 4000):
        b2 = BloomFilter(m, k)
        for i in range(load):
            b2.add(f"x:{i}")
        obs = sum(1 for i in range(10000) if f"y:{i}" in b2) / 10000
        print(f"  {load:>8}{false_positive_rate(m, load, k):>10.4f}{obs:>10.4f}")
    print("\n  Web caches, spell checkers, and databases use one as a fast pre-filter: skip the")
    print("  expensive lookup for anything the filter says is definitely not there.")

    _svg(os.path.join(outdir, "bloom.svg"), m)
    print(f"\n  wrote {os.path.join(outdir, 'bloom.svg')}")


def _svg(path, m, w=760, h=390):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="monospace">',
        f'<rect width="{w}" height="{h}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Bloom filter: false positives rise with load, no false negatives</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'false-positive rate vs items inserted (left); optimal number of hashes (right)</text>',
    ]

    # left: FP rate vs n (theory curve + observed points), fixed m, k
    n_opt = 1000
    k = optimal_num_hashes(m, n_opt)
    lx0, lx1 = 60, w // 2 - 20
    ly0, ly1 = h - 55, 62
    nmax = 5000

    def LX(nn):
        return lx0 + nn / nmax * (lx1 - lx0)

    def LY(p):
        return ly0 - p * (ly0 - ly1)

    parts.append(f'<line x1="{lx0}" y1="{ly0}" x2="{lx1}" y2="{ly0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{lx0}" y1="{ly0}" x2="{lx0}" y2="{ly1}" stroke="#8b949e" stroke-width="1.2"/>')
    # theory curve
    pts = []
    for i in range(0, 101):
        nn = nmax * i / 100
        pts.append(f"{LX(nn):.1f},{LY(false_positive_rate(m, nn, k)):.1f}")
    parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#4dabf7" stroke-width="2.5"/>')
    # observed points
    for load in (250, 500, 1000, 2000, 4000):
        b2 = BloomFilter(m, k)
        for i in range(load):
            b2.add(f"x:{i}")
        obs = sum(1 for i in range(6000) if f"y:{i}" in b2) / 6000
        parts.append(f'<circle cx="{LX(load):.1f}" cy="{LY(obs):.1f}" r="3" fill="#ffd43b"/>')
    # mark design point n=1000 @ ~1%
    parts.append(f'<line x1="{LX(1000):.1f}" y1="{ly1}" x2="{LX(1000):.1f}" y2="{ly0}" '
                 f'stroke="#06d6a0" stroke-width="1" stroke-dasharray="3 3"/>')
    parts.append(f'<text x="{LX(1000)+4:.1f}" y="{ly1+10:.1f}" fill="#06d6a0" font-size="9">'
                 f'design: 1000 @ 1%</text>')
    for p in (0.0, 0.5, 1.0):
        parts.append(f'<text x="{lx0-6:.1f}" y="{LY(p)+3:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="end">{p:.1f}</text>')
    for nn in (0, 2500, 5000):
        parts.append(f'<text x="{LX(nn):.1f}" y="{ly0+15:.1f}" fill="#8b949e" font-size="8" '
                     f'text-anchor="middle">{nn}</text>')
    parts.append(f'<text x="{(lx0+lx1)/2:.1f}" y="{ly0+30:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">items inserted</text>')
    parts.append(f'<rect x="{lx0+8}" y="{ly1}" width="9" height="9" fill="#4dabf7"/>'
                 f'<text x="{lx0+21}" y="{ly1+8}" fill="#e6edf3" font-size="9">theory</text>')
    parts.append(f'<circle cx="{lx0+13}" cy="{ly1+20}" r="4" fill="#ffd43b"/>'
                 f'<text x="{lx0+21}" y="{ly1+23}" fill="#e6edf3" font-size="9">observed</text>')

    # right: FP rate vs number of hashes k (fixed m, n) -- a minimum at the optimal k
    rx0, rx1 = w // 2 + 45, w - 30
    ry0, ry1 = h - 55, 62
    n_fix = 1000
    ks = list(range(1, 16))
    fps = [false_positive_rate(m, n_fix, kk) for kk in ks]
    fmax = max(fps)

    def RX(kk):
        return rx0 + (kk - ks[0]) / (ks[-1] - ks[0]) * (rx1 - rx0)

    def RY(p):
        return ry0 - p / fmax * (ry0 - ry1)

    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx1}" y2="{ry0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx0}" y2="{ry1}" stroke="#8b949e" stroke-width="1.2"/>')
    pts = " ".join(f"{RX(kk):.1f},{RY(p):.1f}" for kk, p in zip(ks, fps))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#8338ec" stroke-width="2.5"/>')
    kbest = ks[min(range(len(fps)), key=lambda i: fps[i])]
    parts.append(f'<circle cx="{RX(kbest):.1f}" cy="{RY(min(fps)):.1f}" r="3.5" fill="#06d6a0"/>')
    parts.append(f'<text x="{RX(kbest):.1f}" y="{RY(min(fps))-7:.1f}" fill="#06d6a0" font-size="9" '
                 f'text-anchor="middle">optimal k = {optimal_num_hashes(m, n_fix)}</text>')
    for kk in (1, 7, 15):
        parts.append(f'<text x="{RX(kk):.1f}" y="{ry0+15:.1f}" fill="#8b949e" font-size="8" '
                     f'text-anchor="middle">{kk}</text>')
    parts.append(f'<text x="{(rx0+rx1)/2:.1f}" y="{ry0+30:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">number of hash functions k</text>')
    parts.append(f'<text x="{rx0+4:.1f}" y="{ry1-2:.1f}" fill="#8b949e" font-size="9">FP rate</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
