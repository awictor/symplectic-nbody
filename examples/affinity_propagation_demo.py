"""Affinity propagation demo: cluster blobs without choosing k, and watch preference tune the cluster count."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import affinity_propagation as ap


PALETTE = {
    "bg": "#0d1117", "blue": "#4dabf7", "yellow": "#ffd43b", "red": "#ff6b6b",
    "green": "#06d6a0", "purple": "#b197fc", "gray": "#8b949e", "text": "#e6edf3",
}
CLUSTER_COLORS = ["#4dabf7", "#ffd43b", "#06d6a0", "#b197fc", "#ff6b6b", "#f783ac", "#63e6be", "#ffa94d"]


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
    lines.append("Affinity propagation -- clustering that elects its own exemplars, no k")
    lines.append("=" * 70)
    lines.append("")

    rng = _R(2)
    centers = [(0.0, 0.0), (9.0, 1.0), (4.0, 8.0), (11.0, 9.0)]
    per = 14
    pts = []
    for ctr in centers:
        for _ in range(per):
            pts.append([ctr[d] + 0.6 * rng.normal() for d in range(2)])

    res = ap.cluster_points(pts)
    lines.append(f"{len(pts)} points from {len(centers)} true blobs, k NOT specified.")
    lines.append(f"Affinity propagation found {res['n_clusters']} clusters "
                 f"in {res['iterations']} iterations.")
    lines.append(f"Exemplars (elected cluster centers): points {res['exemplars']}")
    lines.append("")
    lines.append("Exemplar coordinates:")
    for c, ep in enumerate(res["exemplar_points"]):
        lines.append(f"  cluster {c}: ({ep[0]:+.2f}, {ep[1]:+.2f})")
    lines.append("")

    # preference sweep -> cluster count (the one knob)
    S = ap.negative_sq_euclidean(pts)
    med = ap.median_offdiagonal(S)
    lines.append("Preference (self-similarity) tunes the cluster count -- the only knob:")
    lines.append("   preference       #clusters")
    lines.append("   " + "-" * 30)
    for mult in (5.0, 2.0, 1.0, 0.3, 0.05):
        r = ap.affinity_propagation(S, preference=med * mult, damping=0.7, max_iter=400)
        lines.append(f"   {med * mult:11.1f}       {r['n_clusters']}")
    lines.append("  (less-negative preference makes each point cheaper to elect -> more clusters)")

    text = "\n".join(lines)
    print(text)

    svg = _svg(pts, res)
    return text, svg


def _svg(pts, res):
    W, H = 640, 430
    P = PALETTE
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    px_pad = 0.1 * (xmax - xmin)
    py_pad = 0.1 * (ymax - ymin)
    xmin -= px_pad
    xmax += px_pad
    ymin -= py_pad
    ymax += py_pad

    x0, x1, y0, y1 = 45, 610, 55, 380

    def px(x):
        return x0 + (x - xmin) / (xmax - xmin) * (x1 - x0)

    def py(y):
        return y1 - (y - ymin) / (ymax - ymin) * (y1 - y0)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
    parts.append(f'<rect width="{W}" height="{H}" fill="{P["bg"]}"/>')
    parts.append(f'<text x="20" y="26" fill="{P["text"]}" font-size="15">'
                 f'Affinity propagation: {res["n_clusters"]} exemplars elected by message passing</text>')

    ex_to_c = {ex: c for c, ex in enumerate(res["exemplars"])}
    # draw links from each point to its exemplar
    for i in range(len(pts)):
        ex = res["labels"][i]
        c = ex_to_c[ex]
        col = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
        parts.append(f'<line x1="{px(pts[i][0]):.1f}" y1="{py(pts[i][1]):.1f}" '
                     f'x2="{px(pts[ex][0]):.1f}" y2="{py(pts[ex][1]):.1f}" '
                     f'stroke="{col}" stroke-width="0.6" opacity="0.35"/>')
    # points
    for i in range(len(pts)):
        c = ex_to_c[res["labels"][i]]
        col = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
        parts.append(f'<circle cx="{px(pts[i][0]):.1f}" cy="{py(pts[i][1]):.1f}" r="3" fill="{col}"/>')
    # exemplars as big ringed markers
    for ex in res["exemplars"]:
        c = ex_to_c[ex]
        col = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
        parts.append(f'<circle cx="{px(pts[ex][0]):.1f}" cy="{py(pts[ex][1]):.1f}" r="7" '
                     f'fill="none" stroke="{col}" stroke-width="2.5"/>')
        parts.append(f'<circle cx="{px(pts[ex][0]):.1f}" cy="{py(pts[ex][1]):.1f}" r="2" '
                     f'fill="{P["text"]}"/>')

    parts.append(f'<text x="20" y="{H - 12}" fill="{P["gray"]}" font-size="11">'
                 f'ringed points = exemplars (real data points); thin links = each point to its '
                 f'chosen exemplar. No k was set.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
