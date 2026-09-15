"""Bak-Sneppen demo: evolution self-organizing to a critical threshold, with punctuated-equilibrium avalanches."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import bak_sneppen as bs


PALETTE = {
    "bg": "#0d1117", "blue": "#4dabf7", "yellow": "#ffd43b", "red": "#ff6b6b",
    "green": "#06d6a0", "purple": "#b197fc", "gray": "#8b949e", "text": "#e6edf3",
}


def main():
    lines = []
    lines.append("Bak-Sneppen model -- evolution self-organizes to the edge of chaos")
    lines.append("=" * 66)
    lines.append("")
    lines.append("N species on a ring; each step the LEAST-fit species and its 2 neighbours get")
    lines.append("fresh random fitness. No tuning -- the system finds its own critical state.")
    lines.append("")

    res = bs.simulate(200, 400000, seed=1, track=True)
    lines.append(f"Ring of 200 species, 400k updates:")
    lines.append(f"  self-organized threshold (gap): {res['threshold']:.3f}  (1-D theory f_c ~ 0.667)")
    lines.append(f"  final mean fitness: {res['mean_fitness']:.3f}")
    lines.append(f"  fraction above f_c: {bs.fraction_above(res['fitness'], 0.667):.2f}")
    lines.append("")

    # gap climbing over time
    hist = res["min_history"]
    lines.append("The gap (running max of the minimum fitness) climbs to f_c and plateaus:")
    lines.append("   updates      gap so far")
    lines.append("   " + "-" * 26)
    running = 0.0
    for mark in (10, 100, 1000, 10000, 100000, 400000):
        running = max(hist[:mark])
        lines.append(f"   {mark:7d}      {running:.3f}")
    lines.append("")

    # avalanche distribution
    lines.append("Avalanche sizes (activity below a subcritical threshold) follow a power law:")
    lines.append("   size bin      count")
    lines.append("   " + "-" * 26)
    sizes = bs.avalanche_sizes(200, 400000, threshold=0.5, seed=1)
    bins = [(1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 100000)]
    for lo, hi in bins:
        c = sum(1 for s in sizes if lo <= s < hi)
        label = f"{lo}-{hi-1}" if hi < 100000 else f"{lo}+"
        bar = "#" * min(50, int(40 * c / max(1, len(sizes)) * 3))
        lines.append(f"   {label:10s}   {c:5d} {bar}")
    lines.append("")
    lines.append(f"{len(sizes)} avalanches, sizes 1..{max(sizes)}: many tiny, a few huge -- scale-free.")

    text = "\n".join(lines)
    print(text)

    svg = _svg()
    return text, svg


def _svg():
    W, H = 640, 440
    P = PALETTE
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
    parts.append(f'<rect width="{W}" height="{H}" fill="{P["bg"]}"/>')
    parts.append(f'<text x="20" y="24" fill="{P["text"]}" font-size="15">'
                 f'The minimum fitness (top) and avalanche-size distribution (bottom)</text>')

    # TOP: minimum fitness over time with the self-organized threshold
    res = bs.simulate(150, 60000, seed=3, track=True)
    hist = res["min_history"]
    tx0, tx1, ty0, ty1 = 55, 610, 50, 210
    n = len(hist)
    step = max(1, n // 1000)

    def tx(i):
        return tx0 + i / n * (tx1 - tx0)

    def ty(f):
        return ty1 - f * (ty1 - ty0)

    # f_c line
    fc = bs.critical_threshold_1d()
    parts.append(f'<line x1="{tx0}" y1="{ty(fc):.1f}" x2="{tx1}" y2="{ty(fc):.1f}" '
                 f'stroke="{P["green"]}" stroke-width="1.5" stroke-dasharray="5,4"/>')
    parts.append(f'<text x="{tx1 - 90}" y="{ty(fc) - 4:.1f}" fill="{P["green"]}" font-size="10">'
                 f'f_c ~ 0.667</text>')
    pts = " ".join(f"{tx(i):.1f},{ty(hist[i]):.1f}" for i in range(0, n, step))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="{P["yellow"]}" stroke-width="0.6" opacity="0.8"/>')
    parts.append(f'<line x1="{tx0}" y1="{ty1}" x2="{tx1}" y2="{ty1}" stroke="{P["gray"]}" stroke-width="1"/>')
    parts.append(f'<text x="{tx0}" y="{ty0 - 4:.1f}" fill="{P["gray"]}" font-size="10">'
                 f'least-fit species over time: spikes up to f_c, avalanches drag it down (punctuated)</text>')

    # BOTTOM: avalanche size histogram on log-log
    sizes = bs.avalanche_sizes(200, 300000, threshold=0.5, seed=1)
    bx0, bx1, by0, by1 = 55, 610, 260, 400
    # log-binned counts
    import math as m
    maxs = max(sizes)
    nbins = 12
    counts = [0] * nbins
    for s in sizes:
        b = min(nbins - 1, int(m.log(s) / m.log(maxs + 1) * nbins))
        counts[b] += 1
    cmax = max(counts)

    def bx(i):
        return bx0 + i / nbins * (bx1 - bx0)

    def by(c):
        if c <= 0:
            return by1
        return by1 - (m.log(c + 1) / m.log(cmax + 1)) * (by1 - by0)

    for i in range(nbins):
        h = by1 - by(counts[i])
        parts.append(f'<rect x="{bx(i):.1f}" y="{by(counts[i]):.1f}" width="{(bx1-bx0)/nbins - 1:.1f}" '
                     f'height="{h:.1f}" fill="{P["red"]}" opacity="0.7"/>')
    parts.append(f'<line x1="{bx0}" y1="{by1}" x2="{bx1}" y2="{by1}" stroke="{P["gray"]}" stroke-width="1"/>')
    parts.append(f'<text x="{bx0}" y="{by0 - 6:.1f}" fill="{P["gray"]}" font-size="10">'
                 f'avalanche size (log bins, log count): a straight-ish falloff = power law</text>')

    parts.append(f'<text x="20" y="{H - 10}" fill="{P["gray"]}" font-size="11">'
                 f'no parameter is tuned -- the ecosystem drives ITSELF to criticality, producing '
                 f'extinction bursts of every size.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
