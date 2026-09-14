"""Fuzzy c-means demo: soft memberships shown as color blends, with the fuzzifier sweep and boundary points."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import fuzzy_cmeans as fcm


PALETTE = {
    "bg": "#0d1117", "blue": "#4dabf7", "yellow": "#ffd43b", "red": "#ff6b6b",
    "green": "#06d6a0", "purple": "#b197fc", "gray": "#8b949e", "text": "#e6edf3",
}
# three cluster base colors (RGB) for membership blending
BASE = [(77, 171, 247), (255, 107, 107), (6, 214, 160)]


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
    lines.append("Fuzzy c-means -- soft clustering by graded membership")
    lines.append("=" * 54)
    lines.append("")

    # three overlapping blobs so boundary points are genuinely ambiguous
    rng = _R(3)
    truth = [(0.0, 0.0), (6.0, 0.5), (3.0, 5.0)]
    data = []
    for ctr in truth:
        for _ in range(30):
            data.append([ctr[d] + 1.3 * rng.normal() for d in range(2)])

    res = fcm.fuzzy_cmeans(data, c=3, m=2.0, seed=1, track=True)
    lines.append(f"{len(data)} points, 3 overlapping blobs, fuzzifier m=2.0.")
    lines.append(f"Converged in {res['iterations']} iterations, objective {res['objective']:.1f}.")
    lines.append(f"Partition coefficient: {res['partition_coefficient']:.3f}  (1=crisp, 1/3=maximally fuzzy)")
    lines.append("")
    lines.append("Recovered centers:")
    for j, c in enumerate(res["centers"]):
        lines.append(f"  cluster {j}: ({c[0]:+.2f}, {c[1]:+.2f})")
    lines.append("")

    # count ambiguous points (max membership < 0.6)
    ambiguous = sum(1 for row in res["memberships"] if max(row) < 0.6)
    lines.append(f"Ambiguous points (no cluster > 60% membership): {ambiguous}/{len(data)}")
    lines.append("  -- these sit in the overlaps; hard k-means would force an arbitrary choice.")
    lines.append("")

    # fuzzifier sweep -> partition coefficient
    lines.append("Fuzzifier m controls softness (partition coefficient):")
    lines.append("   m       partition coeff   mean max-membership")
    lines.append("   " + "-" * 42)
    for m in (1.1, 1.5, 2.0, 3.0, 5.0):
        r = fcm.fuzzy_cmeans(data, c=3, m=m, seed=1)
        mean_max = sum(max(row) for row in r["memberships"]) / len(data)
        lines.append(f"   {m:4.1f}    {r['partition_coefficient']:.3f}             {mean_max:.3f}")
    lines.append("  (larger m -> softer memberships -> lower partition coefficient)")

    text = "\n".join(lines)
    print(text)

    svg = _svg(data, res)
    return text, svg


def _svg(data, res):
    W, H = 640, 430
    P = PALETTE
    xs = [p[0] for p in data]
    ys = [p[1] for p in data]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    padx = 0.1 * (xmax - xmin)
    pady = 0.1 * (ymax - ymin)
    xmin -= padx
    xmax += padx
    ymin -= pady
    ymax += pady

    x0, x1, y0, y1 = 40, 610, 55, 375

    def px(x):
        return x0 + (x - xmin) / (xmax - xmin) * (x1 - x0)

    def py(y):
        return y1 - (y - ymin) / (ymax - ymin) * (y1 - y0)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
    parts.append(f'<rect width="{W}" height="{H}" fill="{P["bg"]}"/>')
    parts.append(f'<text x="20" y="26" fill="{P["text"]}" font-size="15">'
                 f'Fuzzy c-means: point color = blend of its cluster memberships</text>')

    # each point colored by membership-weighted blend of the three base colors
    for i, p in enumerate(data):
        mem = res["memberships"][i]
        r = sum(mem[j] * BASE[j][0] for j in range(3))
        g = sum(mem[j] * BASE[j][1] for j in range(3))
        b = sum(mem[j] * BASE[j][2] for j in range(3))
        col = f"#{int(r):02x}{int(g):02x}{int(b):02x}"
        # radius grows with ambiguity (low max membership = bigger, to highlight the fuzzy ones)
        amb = 1.0 - max(mem)
        rad = 2.5 + 4.0 * amb
        parts.append(f'<circle cx="{px(p[0]):.1f}" cy="{py(p[1]):.1f}" r="{rad:.1f}" '
                     f'fill="{col}" opacity="0.85"/>')

    # centers as white crosses
    for c in res["centers"]:
        cx, cy = px(c[0]), py(c[1])
        parts.append(f'<line x1="{cx-6:.1f}" y1="{cy:.1f}" x2="{cx+6:.1f}" y2="{cy:.1f}" '
                     f'stroke="{P["text"]}" stroke-width="2.5"/>')
        parts.append(f'<line x1="{cx:.1f}" y1="{cy-6:.1f}" x2="{cx:.1f}" y2="{cy+6:.1f}" '
                     f'stroke="{P["text"]}" stroke-width="2.5"/>')

    parts.append(f'<text x="20" y="{H - 26}" fill="{P["gray"]}" font-size="11">'
                 f'pure blue/red/green = crisp membership; blended/larger dots = ambiguous points in '
                 f'the overlaps.</text>')
    parts.append(f'<text x="20" y="{H - 10}" fill="{P["gray"]}" font-size="11">'
                 f'white crosses = cluster centers. Hard clustering would erase the gradient.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
