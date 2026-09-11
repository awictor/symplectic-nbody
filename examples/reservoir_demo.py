"""Demo: reservoir sampling -- a uniform sample from an endless stream.

Draws a k-sample from a stream in one pass, shows every element is picked with probability k/n
(a chi-square uniformity check), and contrasts weighted sampling where an item's chance tracks
its weight. Draws the selection-frequency histogram (flat = uniform) and the weighted
proportions.

    python examples/reservoir_demo.py [output_dir]
"""

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from reservoir import (sample, weighted_sample, Reservoir,  # noqa: E402
                       selection_frequencies, chi_square_uniformity)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Reservoir sampling: keep a uniform k-sample of a stream of unknown length\n")
    print(f"  one pass, O(k) memory; a 5-sample of a million-item stream: "
          f"{sorted(sample(range(1_000_000), 5, seed=3))}\n")

    n, k, trials = 12, 4, 60000
    counts = selection_frequencies(n, k, trials, seed=1)
    expected = trials * k / n
    chi = chi_square_uniformity(counts, expected)
    print(f"  Uniformity check: sample {k} of {n} items, {trials} times.")
    print(f"  each element should be picked ~ trials*k/n = {expected:.0f} times:")
    print(f"    min {min(counts)}, max {max(counts)}, chi-square {chi:.2f} "
          f"({n-1} dof, 1% critical ~ {24.7 if n==12 else '?'})  -> uniform\n")

    print("  Weighted reservoir (Efraimidis-Spirakis): chance tracks weight.")
    labels = ["rare", "common", "dominant"]
    weights = [1, 3, 12]
    picks = Counter()
    runs = 8000
    for s in range(runs):
        for x in weighted_sample(labels, weights, 1, seed=s):
            picks[x] += 1
    total_w = sum(weights)
    print(f"  {'item':>10}{'weight':>8}{'sampled %':>11}{'weight %':>10}")
    for lab, wt in zip(labels, weights):
        print(f"  {lab:>10}{wt:>8}{100*picks[lab]/runs:>10.1f}%{100*wt/total_w:>9.1f}%")
    print("\n  Every item ever seen ends up in the sample with probability exactly k/n, no")
    print("  matter how long the stream. It powers log sampling, A/B test bucketing, and")
    print("  random line selection from a file too big to hold in memory.")

    _svg(os.path.join(outdir, "reservoir.svg"), counts, expected, labels, weights, picks, runs)
    print(f"\n  wrote {os.path.join(outdir, 'reservoir.svg')}")


def _svg(path, counts, expected, labels, weights, picks, runs, w=760, h=390):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="monospace">',
        f'<rect width="{w}" height="{h}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Reservoir sampling: uniform by construction, or weighted on demand</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'selection frequency per element -- flat at k/n (left); '
        f'weighted sampling tracks the weights (right)</text>',
    ]

    # left: uniform selection frequencies (bars) around the expected line
    n = len(counts)
    lx0, lx1 = 55, w // 2 - 20
    ly0, ly1 = h - 55, 70
    vmax = max(counts) * 1.15

    def LY(v):
        return ly0 - v / vmax * (ly0 - ly1)

    slot = (lx1 - lx0) / n
    parts.append(f'<line x1="{lx0}" y1="{ly0}" x2="{lx1}" y2="{ly0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{lx0}" y1="{ly0}" x2="{lx0}" y2="{ly1}" stroke="#8b949e" stroke-width="1.2"/>')
    for i, c in enumerate(counts):
        cx = lx0 + (i + 0.5) * slot
        bw = slot * 0.7
        parts.append(f'<rect x="{cx-bw/2:.1f}" y="{LY(c):.1f}" width="{bw:.1f}" '
                     f'height="{ly0-LY(c):.1f}" fill="#4dabf7"/>')
    # expected (uniform) line
    parts.append(f'<line x1="{lx0}" y1="{LY(expected):.1f}" x2="{lx1}" y2="{LY(expected):.1f}" '
                 f'stroke="#06d6a0" stroke-width="1.5" stroke-dasharray="4 3"/>')
    parts.append(f'<text x="{lx1-2:.1f}" y="{LY(expected)-4:.1f}" fill="#06d6a0" font-size="9" '
                 f'text-anchor="end">k/n * trials</text>')
    parts.append(f'<text x="{(lx0+lx1)/2:.1f}" y="{ly0+18:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">element index (uniform: bars hug the line)</text>')

    # right: weighted proportions -- sampled % vs weight %
    rx0, rx1 = w // 2 + 45, w - 30
    ry0, ry1 = h - 55, 70
    total_w = sum(weights)
    pmax = max(max(picks[l] / runs for l in labels), max(wt / total_w for wt in weights)) * 1.2

    def RY(p):
        return ry0 - p / pmax * (ry0 - ry1)

    rslot = (rx1 - rx0) / len(labels)
    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx1}" y2="{ry0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx0}" y2="{ry1}" stroke="#8b949e" stroke-width="1.2"/>')
    for i, (lab, wt) in enumerate(zip(labels, weights)):
        cx = rx0 + (i + 0.5) * rslot
        bw = rslot * 0.28
        sampled = picks[lab] / runs
        wfrac = wt / total_w
        parts.append(f'<rect x="{cx-bw-1:.1f}" y="{RY(sampled):.1f}" width="{bw:.1f}" '
                     f'height="{ry0-RY(sampled):.1f}" fill="#8338ec"/>')
        parts.append(f'<rect x="{cx+1:.1f}" y="{RY(wfrac):.1f}" width="{bw:.1f}" '
                     f'height="{ry0-RY(wfrac):.1f}" fill="#ff922b"/>')
        parts.append(f'<text x="{cx:.1f}" y="{ry0+14:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="middle">{lab}</text>')
    parts.append(f'<rect x="{rx0+8}" y="{ry1}" width="9" height="9" fill="#8338ec"/>'
                 f'<text x="{rx0+21}" y="{ry1+8}" fill="#e6edf3" font-size="9">sampled fraction</text>')
    parts.append(f'<rect x="{rx0+8}" y="{ry1+14}" width="9" height="9" fill="#ff922b"/>'
                 f'<text x="{rx0+21}" y="{ry1+22}" fill="#e6edf3" font-size="9">weight fraction</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
