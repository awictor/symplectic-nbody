"""Demo: the Abelian sandpile -- self-organized criticality.

Prints the topple count for a tall central stack and the avalanche statistics of a
self-organized pile, then draws the relaxed stack pattern (grain heights coloured) and the
heavy-tailed avalanche-size distribution on log-log axes.

    python examples/sandpile_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sandpile import relax_stack, avalanche_series, total_grains  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Abelian sandpile: pile topples at 4 grains, sending one to each neighbour\n")
    print("  Relaxing a central stack (no parameter tuning -> self-organized pattern):")
    for h in (100, 500, 1000):
        grid, topples = relax_stack(21, h)
        print(f"    {h:>5} grains  ->  {topples:>6} topples, {total_grains(grid)} left on grid")

    print("\n  Avalanche statistics from 8000 random drops (20x20):")
    sizes = avalanche_series(20, 8000, seed=1)
    settled = sizes[4000:]                 # after the pile self-organizes
    nonzero = [s for s in settled if s > 0]
    print(f"    avalanches with topples: {len(nonzero)} of {len(settled)}")
    print(f"    mean size {sum(settled)/len(settled):.1f}, max size {max(settled)}")
    print("    (mostly small, with rare system-spanning cascades -- a power-law tail)")

    print("\n  With no tuning, the pile drives itself to a critical state where one grain can")
    print("  trigger an avalanche of any size, scale-free like earthquakes, forest fires, and")
    print("  neuronal cascades. The founding model of self-organized criticality.")

    _svg(os.path.join(outdir, "sandpile.svg"), sizes)
    print(f"\n  wrote {os.path.join(outdir, 'sandpile.svg')}")


def _svg(path, sizes, size=720, pad=64):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">The Abelian sandpile</text>',
        f'<text x="20" y="46" fill="#8b949e" font-size="12">'
        f'a relaxed central stack (left); the heavy-tailed avalanche-size distribution (right)</text>',
    ]

    # --- left: relaxed stack pattern ---
    grid, _ = relax_stack(41, 2000)
    n = len(grid)
    lx0 = pad
    panel = size * 0.44
    cell = panel / n
    top = 60
    colors = ["#0d1117", "#1b3a6b", "#4dabf7", "#ffd43b"]   # 0,1,2,3 grains
    for r in range(n):
        for c in range(n):
            v = min(3, grid[r][c])
            if v == 0:
                continue
            parts.append(f'<rect x="{lx0 + c*cell:.1f}" y="{top + r*cell:.1f}" '
                         f'width="{cell+0.5:.1f}" height="{cell+0.5:.1f}" fill="{colors[v]}"/>')
    parts.append(f'<text x="{lx0 + panel/2:.1f}" y="{top + panel + 18:.1f}" fill="#8b949e" '
                 f'font-size="10" text-anchor="middle">relaxed stack (colour = grains 1-3): self-similar</text>')

    # --- right: avalanche-size distribution, log-log ---
    settled = [s for s in sizes[len(sizes)//2:] if s > 0]
    # histogram in log-spaced bins
    if settled:
        smax = max(settled)
        nbins = 20
        import math as _m
        counts = [0] * nbins
        lo = _m.log10(1)
        hi = _m.log10(smax + 1)
        for s in settled:
            b = int((_m.log10(s) - lo) / (hi - lo + 1e-9) * (nbins - 1))
            counts[max(0, min(nbins - 1, b))] += 1
    else:
        counts = [1]
        hi = 1.0
        lo = 0.0

    rx0, rx1 = size * 0.56, size - pad
    ry0, ry1 = size - pad, 90
    cmax = max(counts) or 1
    def BX(b):
        return rx0 + b / (len(counts) - 1) * (rx1 - rx0)
    def CY(c):
        import math as _m
        v = _m.log10(c + 1)
        vmax = _m.log10(cmax + 1)
        return ry0 - v / vmax * (ry0 - ry1)
    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx1}" y2="{ry0}" stroke="#8b949e" stroke-width="1.4"/>')
    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx0}" y2="{ry1}" stroke="#8b949e" stroke-width="1.4"/>')
    bw = (rx1 - rx0) / len(counts) * 0.8
    for b, c in enumerate(counts):
        if c == 0:
            continue
        parts.append(f'<rect x="{BX(b)-bw/2:.1f}" y="{CY(c):.1f}" width="{bw:.1f}" '
                     f'height="{ry0-CY(c):.1f}" fill="#ff922b" opacity="0.85"/>')
    parts.append(f'<text x="{(rx0+rx1)/2:.1f}" y="{ry0+16:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">avalanche size (log bins) -> count (log)</text>')
    parts.append(f'<text x="{rx0+8:.1f}" y="{ry1-4:.1f}" fill="#ff922b" font-size="10">'
                 f'straight-ish decline on log-log = power-law tail</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
