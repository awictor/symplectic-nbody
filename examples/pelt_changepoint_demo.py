"""PELT demo: optimal multi-changepoint segmentation of a noisy piecewise signal, with the penalty tradeoff."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import pelt_changepoint as pc


PALETTE = {
    "bg": "#0d1117", "blue": "#4dabf7", "yellow": "#ffd43b", "red": "#ff6b6b",
    "green": "#06d6a0", "purple": "#b197fc", "gray": "#8b949e", "text": "#e6edf3",
}


class _R:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF
        self._spare = None

    def u(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return (self.s >> 8) / (1 << 24)

    def normal(self):
        if self._spare is not None:
            v = self._spare
            self._spare = None
            return v
        u1 = max(self.u(), 1e-12)
        u2 = self.u()
        r = math.sqrt(-2 * math.log(u1))
        self._spare = r * math.sin(2 * math.pi * u2)
        return r * math.cos(2 * math.pi * u2)


def main():
    lines = []
    lines.append("PELT -- Pruned Exact Linear Time optimal change-point detection")
    lines.append("=" * 64)
    lines.append("")

    # piecewise-constant signal with 4 segments
    rng = _R(3)
    segments = [(2.0, 50), (7.0, 40), (3.0, 60), (9.0, 50)]
    data = []
    truth = []
    pos = 0
    for level, length in segments:
        for _ in range(length):
            data.append(level + 0.8 * rng.normal())
        pos += length
        truth.append(pos)
    truth = truth[:-1]

    res = pc.pelt(data)
    lines.append(f"Signal of {len(data)} points, {len(segments)} true levels, noise sd 0.8.")
    lines.append(f"True change points: {truth}")
    lines.append(f"PELT change points: {res['change_points']}  (penalty = {res['penalty']:.2f} = 2 log n)")
    lines.append(f"Segments found: {res['n_segments']}")
    lines.append(f"Segment means: {[round(m, 2) for m in res['segment_means']]}")
    lines.append(f"True levels:   {[s[0] for s in segments]}")
    lines.append("")

    # verify pruning matches the exact O(n^2) DP
    brute = pc.pelt_bruteforce(data)
    lines.append(f"Exact O(n^2) DP change points: {brute}")
    lines.append(f"  PELT (pruned, ~O(n)) gives the SAME optimum: {res['change_points'] == brute}")
    lines.append("")

    # penalty sweep
    lines.append("Penalty controls the fit-vs-parsimony tradeoff:")
    lines.append("   penalty   #change points   segments")
    lines.append("   " + "-" * 38)
    for pen in (0.5, 2.0, res["penalty"], 20.0, 100.0, 1e6):
        r = pc.pelt(data, penalty=pen)
        tag = "  <- 2 log n" if abs(pen - res["penalty"]) < 1e-9 else ""
        lines.append(f"   {pen:8.2f}   {len(r['change_points']):8d}       {r['n_segments']}{tag}")
    lines.append("")
    lines.append("Too small over-segments (change point at every wiggle); too large misses real shifts.")

    text = "\n".join(lines)
    print(text)

    svg = _svg(data, res, truth)
    return text, svg


def _svg(data, res, truth):
    W, H = 640, 420
    P = PALETTE
    n = len(data)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
    parts.append(f'<rect width="{W}" height="{H}" fill="{P["bg"]}"/>')
    parts.append(f'<text x="20" y="26" fill="{P["text"]}" font-size="15">'
                 f'PELT: {res["n_segments"]} optimal segments of a noisy step signal</text>')

    x0, x1, y0, y1 = 45, 615, 55, 350
    vmin, vmax = min(data), max(data)
    pad = 0.08 * (vmax - vmin)
    vmin -= pad
    vmax += pad

    def px(i):
        return x0 + i / (n - 1) * (x1 - x0)

    def py(v):
        return y1 - (v - vmin) / (vmax - vmin) * (y1 - y0)

    # true change points (gray dashed)
    for t in truth:
        parts.append(f'<line x1="{px(t):.1f}" y1="{y0}" x2="{px(t):.1f}" y2="{y1}" '
                     f'stroke="#30363d" stroke-width="1" stroke-dasharray="3,3"/>')

    # raw data
    pts = " ".join(f"{px(i):.1f},{py(data[i]):.1f}" for i in range(n))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="{P["blue"]}" stroke-width="0.8" opacity="0.6"/>')

    # fitted piecewise-constant means (yellow steps)
    bounds = [0] + res["change_points"] + [n]
    means = res["segment_means"]
    for i in range(len(means)):
        a, b = bounds[i], bounds[i + 1]
        parts.append(f'<line x1="{px(a):.1f}" y1="{py(means[i]):.1f}" x2="{px(b - 1):.1f}" '
                     f'y2="{py(means[i]):.1f}" stroke="{P["yellow"]}" stroke-width="2.5"/>')
    # change-point verticals (red)
    for c in res["change_points"]:
        parts.append(f'<line x1="{px(c):.1f}" y1="{y0}" x2="{px(c):.1f}" y2="{y1}" '
                     f'stroke="{P["red"]}" stroke-width="1.5"/>')

    parts.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="{P["gray"]}" stroke-width="1"/>')
    parts.append(f'<text x="{x0}" y="{y1 + 22:.1f}" fill="{P["blue"]}" font-size="11">blue = noisy data</text>')
    parts.append(f'<text x="{x0 + 150}" y="{y1 + 22:.1f}" fill="{P["yellow"]}" font-size="11">yellow = fitted segment means</text>')
    parts.append(f'<text x="{x0 + 400}" y="{y1 + 22:.1f}" fill="{P["red"]}" font-size="11">red = change points</text>')
    parts.append(f'<text x="20" y="{H - 12}" fill="{P["gray"]}" font-size="11">'
                 f'PELT returns the globally optimal segmentation exactly -- and the pruning matches the '
                 f'O(n^2) DP in ~O(n) time.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
