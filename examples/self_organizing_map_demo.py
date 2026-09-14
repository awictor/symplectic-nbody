"""SOM demo: train a Kohonen grid on colored clusters, show the topology-preserving map and error curves."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import self_organizing_map as som


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
    lines.append("Self-organizing map -- a Kohonen grid that preserves neighborhoods")
    lines.append("=" * 66)
    lines.append("")

    # 3-D RGB-like color clusters
    rng = _R(4)
    named = [("red", (0.9, 0.1, 0.1)), ("green", (0.1, 0.85, 0.2)),
             ("blue", (0.15, 0.2, 0.9)), ("yellow", (0.9, 0.85, 0.15))]
    data = []
    labels = []
    for ci, (nm, ctr) in enumerate(named):
        for _ in range(25):
            data.append([min(1, max(0, ctr[d] + 0.06 * rng.normal())) for d in range(3)])
            labels.append(ci)

    rows, cols = 10, 10
    m = som.SOM(rows, cols, 3, seed=1)
    m, hist = m.train(data, epochs=60, alpha0=0.5, track=True)

    qe = m.quantization_error(data)
    te = m.topographic_error(data)
    lines.append(f"{len(data)} points ({len(named)} color clusters) mapped to a {rows}x{cols} grid.")
    lines.append(f"Quantization error: {hist[0]:.4f} -> {qe:.4f}")
    lines.append(f"Topographic error:  {te:.3f}  (fraction with non-adjacent top-2 BMUs)")
    lines.append("")

    # per-cluster BMU centroid on the grid
    from collections import defaultdict
    coords = defaultdict(list)
    for i, x in enumerate(data):
        coords[labels[i]].append(m.map_point(x))
    lines.append("Each color cluster occupies its own grid region:")
    for ci, (nm, _ctr) in enumerate(named):
        rs = [p[0] for p in coords[ci]]
        cs = [p[1] for p in coords[ci]]
        lines.append(f"  {nm:7s}: grid centroid ({sum(rs)/len(rs):.1f}, {sum(cs)/len(cs):.1f})")
    lines.append("")
    lines.append("Neighboring colors land in neighboring cells -- the map is topology-preserving.")

    text = "\n".join(lines)
    print(text)

    svg = _svg(m, data, labels, named, hist)
    return text, svg


def _svg(m, data, labels, named, hist):
    W, H = 640, 460
    P = PALETTE
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
    parts.append(f'<rect width="{W}" height="{H}" fill="{P["bg"]}"/>')
    parts.append(f'<text x="20" y="26" fill="{P["text"]}" font-size="15">'
                 f'SOM: each grid node painted by its learned weight (RGB)</text>')

    # LEFT: the grid, each cell filled with its weight vector interpreted as RGB
    gx0, gy0 = 40, 55
    cell = 30
    for r in range(m.rows):
        for c in range(m.cols):
            w = m.weights[r][c]
            rr = int(255 * min(1, max(0, w[0])))
            gg = int(255 * min(1, max(0, w[1])))
            bb = int(255 * min(1, max(0, w[2])))
            parts.append(f'<rect x="{gx0 + c * cell}" y="{gy0 + r * cell}" '
                         f'width="{cell - 1}" height="{cell - 1}" '
                         f'fill="#{rr:02x}{gg:02x}{bb:02x}"/>')
    parts.append(f'<text x="{gx0}" y="{gy0 + m.rows * cell + 18}" fill="{P["gray"]}" '
                 f'font-size="11">node weights = smooth color gradient (neighbors similar)</text>')

    # mark cluster BMU centroids
    from collections import defaultdict
    coords = defaultdict(list)
    for i, x in enumerate(data):
        coords[labels[i]].append(m.map_point(x))
    for ci, (nm, _ctr) in enumerate(named):
        rs = [p[0] for p in coords[ci]]
        cs = [p[1] for p in coords[ci]]
        mr = sum(rs) / len(rs)
        mc = sum(cs) / len(cs)
        cx = gx0 + mc * cell + cell / 2
        cy = gy0 + mr * cell + cell / 2
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="none" '
                     f'stroke="{P["text"]}" stroke-width="2"/>')

    # RIGHT: quantization-error curve
    rx0, rx1, ry0, ry1 = 400, 610, 70, 300
    n = len(hist)
    hmax = max(hist)
    hmin = min(hist)

    def px(i):
        return rx0 + i / (n - 1) * (rx1 - rx0)

    def py(v):
        return ry1 - (v - hmin) / (hmax - hmin + 1e-12) * (ry1 - ry0)

    parts.append(f'<text x="{rx0}" y="{ry0 - 8}" fill="{P["gray"]}" font-size="11">'
                 f'quantization error vs epoch</text>')
    parts.append(f'<line x1="{rx0}" y1="{ry1}" x2="{rx1}" y2="{ry1}" stroke="{P["gray"]}" stroke-width="1"/>')
    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx0}" y2="{ry1}" stroke="{P["gray"]}" stroke-width="1"/>')
    pts = " ".join(f"{px(i):.1f},{py(hist[i]):.1f}" for i in range(n))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="{P["yellow"]}" stroke-width="2"/>')
    parts.append(f'<text x="{rx0}" y="{ry1 + 16}" fill="{P["gray"]}" font-size="10">0</text>')
    parts.append(f'<text x="{rx1 - 20}" y="{ry1 + 16}" fill="{P["gray"]}" font-size="10">{n}</text>')

    parts.append(f'<text x="20" y="{H - 12}" fill="{P["gray"]}" font-size="11">'
                 f'white rings = where each color cluster centroids on the grid; error falls as '
                 f'the sheet unfolds to fit the data.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
