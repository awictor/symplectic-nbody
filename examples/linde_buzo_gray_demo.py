"""LBG demo: grow a vector-quantization codebook by splitting, and show the codewords tiling the data."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import linde_buzo_gray as lbg


PALETTE = {
    "bg": "#0d1117", "blue": "#4dabf7", "yellow": "#ffd43b", "red": "#ff6b6b",
    "green": "#06d6a0", "purple": "#b197fc", "gray": "#8b949e", "text": "#e6edf3",
}
COLORS = ["#4dabf7", "#ffd43b", "#06d6a0", "#b197fc", "#ff6b6b", "#f783ac", "#63e6be", "#ffa94d"]


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
    lines.append("Linde-Buzo-Gray -- vector-quantization codebook by splitting + refining")
    lines.append("=" * 72)
    lines.append("")

    # data: a curved manifold (spiral-ish) so VQ has to tile a nontrivial shape
    rng = _R(4)
    data = []
    for i in range(400):
        t = 2.5 * math.pi * i / 400
        r = 0.3 + t / 8.0
        x = r * math.cos(t) + 0.08 * rng.normal()
        y = r * math.sin(t) + 0.08 * rng.normal()
        data.append([x, y])

    res = lbg.design(data, 16, track=True)
    lines.append(f"{len(data)} points on a spiral, codebook grown 1 -> 2 -> 4 -> 8 -> 16.")
    lines.append("")
    lines.append("Distortion as the codebook doubles:")
    lines.append("   size   distortion")
    lines.append("   " + "-" * 22)
    for sz, d in res["split_history"]:
        lines.append(f"   {sz:4d}   {d:.5f}")
    lines.append("")

    # bit rate vs distortion
    lines.append("Rate-distortion (bits per vector = log2 size):")
    lines.append("   bits   size   distortion")
    lines.append("   " + "-" * 30)
    for bits in (1, 2, 3, 4, 5):
        r = lbg.design(data, 2 ** bits)
        lines.append(f"   {bits:4d}   {2**bits:4d}   {r['distortion']:.5f}")
    lines.append("")
    lines.append("Each extra bit halves the codebook spacing -> ~4x lower distortion (1/N per dim).")

    text = "\n".join(lines)
    print(text)

    svg = _svg(data, res["codebook"])
    return text, svg


def _svg(data, codebook):
    W, H = 640, 440
    P = PALETTE
    xs = [p[0] for p in data]
    ys = [p[1] for p in data]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    padx = 0.08 * (xmax - xmin)
    pady = 0.08 * (ymax - ymin)
    xmin -= padx
    xmax += padx
    ymin -= pady
    ymax += pady

    x0, x1, y0, y1 = 40, 610, 55, 400

    def px(x):
        return x0 + (x - xmin) / (xmax - xmin) * (x1 - x0)

    def py(y):
        return y1 - (y - ymin) / (ymax - ymin) * (y1 - y0)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
    parts.append(f'<rect width="{W}" height="{H}" fill="{P["bg"]}"/>')
    parts.append(f'<text x="20" y="26" fill="{P["text"]}" font-size="15">'
                 f'LBG: {len(codebook)} codewords tiling a spiral (each point to nearest)</text>')

    codes = lbg.encode(data, codebook)
    # points colored by their assigned codeword
    for i, p in enumerate(data):
        col = COLORS[codes[i] % len(COLORS)]
        parts.append(f'<circle cx="{px(p[0]):.1f}" cy="{py(p[1]):.1f}" r="2.3" '
                     f'fill="{col}" opacity="0.7"/>')
    # codewords as big white-ringed markers
    for k, c in enumerate(codebook):
        col = COLORS[k % len(COLORS)]
        parts.append(f'<circle cx="{px(c[0]):.1f}" cy="{py(c[1]):.1f}" r="6" '
                     f'fill="{col}" stroke="{P["text"]}" stroke-width="1.5"/>')

    parts.append(f'<text x="20" y="{H - 12}" fill="{P["gray"]}" font-size="11">'
                 f'ringed markers = codebook vectors; each colored region is a Voronoi cell -- the '
                 f'codebook adapts to the manifold shape.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
