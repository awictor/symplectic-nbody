"""Galton-Watson demo: extinction probability as a PGF fixed point, and lineage trajectories exploding or dying."""

import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import galton_watson as gw


PALETTE = {
    "bg": "#0d1117", "blue": "#4dabf7", "yellow": "#ffd43b", "red": "#ff6b6b",
    "green": "#06d6a0", "purple": "#b197fc", "gray": "#8b949e", "text": "#e6edf3",
}


def main():
    lines = []
    lines.append("Galton-Watson branching process -- extinction or explosion?")
    lines.append("=" * 60)
    lines.append("")

    lines.append("Extinction probability q = smallest fixed point of s = G(s):")
    lines.append("   offspring dist         mean m   regime         q (extinction)")
    lines.append("   " + "-" * 62)
    cases = [
        ("p0=.5 p1=.3 p2=.2", [0.5, 0.3, 0.2]),
        ("p0=.25 p1=.5 p2=.25", [0.25, 0.5, 0.25]),
        ("p0=.2 p1=.3 p2=.5", [0.2, 0.3, 0.5]),
        ("p0=.3 p2=.7 (fission)", [0.3, 0.0, 0.7]),
        ("p0=.1 p1=.2 p2=.7", [0.1, 0.2, 0.7]),
    ]
    for name, probs in cases:
        m = gw.offspring_mean(probs)
        q = gw.extinction_probability(probs)
        lines.append(f"   {name:22s}  {m:5.2f}   {gw.classify(probs):13s}  {q:.4f}")
    lines.append("")
    lines.append("Subcritical & critical (m<=1): certain extinction. Supercritical (m>1): q<1, chance of growth.")
    lines.append("")

    # empirical vs theoretical extinction
    probs = [0.2, 0.3, 0.5]
    q = gw.extinction_probability(probs)
    emp = gw.empirical_extinction(probs, n_generations=40, n_runs=4000, seed=1)
    lines.append(f"Supercritical p0=.2 p1=.3 p2=.5 (m=1.3): theoretical q={q:.3f}, "
                 f"empirical extinction={emp:.3f} over 4000 lineages.")
    lines.append("")

    # generation growth
    m = gw.offspring_mean(probs)
    lines.append("Expected generation size E[Z_n] = m^n (supercritical growth):")
    lines.append("   n     m^n")
    lines.append("   " + "-" * 16)
    for n in (0, 2, 4, 6, 8, 10):
        lines.append(f"   {n:2d}    {m**n:8.2f}")

    text = "\n".join(lines)
    print(text)

    svg = _svg(probs, q)
    return text, svg


def _svg(probs, q):
    W, H = 640, 460
    P = PALETTE
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
    parts.append(f'<rect width="{W}" height="{H}" fill="{P["bg"]}"/>')
    parts.append(f'<text x="20" y="24" fill="{P["text"]}" font-size="15">'
                 f'Extinction as a fixed point (top) and lineage fates (bottom)</text>')

    # TOP: G(s) vs s, showing the fixed points
    gx0, gx1, gy0, gy1 = 60, 300, 55, 210

    def gx(s):
        return gx0 + s * (gx1 - gx0)

    def gy(v):
        return gy1 - v * (gy1 - gy0)

    # diagonal y=s
    parts.append(f'<line x1="{gx(0):.1f}" y1="{gy(0):.1f}" x2="{gx(1):.1f}" y2="{gy(1):.1f}" '
                 f'stroke="{P["gray"]}" stroke-width="1" stroke-dasharray="3,3"/>')
    # G(s) curve
    N = 100
    pts = " ".join(f"{gx(i/N):.1f},{gy(gw.pgf(probs, i/N)):.1f}" for i in range(N + 1))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="{P["yellow"]}" stroke-width="2"/>')
    # fixed points: q and 1
    parts.append(f'<circle cx="{gx(q):.1f}" cy="{gy(q):.1f}" r="4" fill="{P["red"]}"/>')
    parts.append(f'<circle cx="{gx(1):.1f}" cy="{gy(1):.1f}" r="4" fill="{P["gray"]}"/>')
    parts.append(f'<text x="{gx(q):.1f}" y="{gy(q) + 18:.1f}" fill="{P["red"]}" font-size="10" '
                 f'text-anchor="middle">q={q:.2f}</text>')
    parts.append(f'<text x="{gx0}" y="{gy0 - 4:.1f}" fill="{P["gray"]}" font-size="10">'
                 f'G(s) meets y=s at q and 1</text>')
    parts.append(f'<rect x="{gx0}" y="{gy0}" width="{gx1-gx0}" height="{gy1-gy0}" fill="none" '
                 f'stroke="{P["gray"]}" stroke-width="1"/>')

    # cobweb from 0 to q
    x = 0.0
    for _ in range(12):
        gxs = gw.pgf(probs, x)
        parts.append(f'<line x1="{gx(x):.1f}" y1="{gy(x):.1f}" x2="{gx(x):.1f}" y2="{gy(gxs):.1f}" '
                     f'stroke="{P["green"]}" stroke-width="0.6" opacity="0.6"/>')
        parts.append(f'<line x1="{gx(x):.1f}" y1="{gy(gxs):.1f}" x2="{gx(gxs):.1f}" y2="{gy(gxs):.1f}" '
                     f'stroke="{P["green"]}" stroke-width="0.6" opacity="0.6"/>')
        x = gxs

    # BOTTOM: lineage trajectories (log-ish population vs generation)
    lx0, lx1, ly0, ly1 = 340, 610, 55, 230
    n_gen = 12
    max_size = 1

    def lx(g):
        return lx0 + g / n_gen * (lx1 - lx0)

    trajs = []
    for r in range(30):
        sizes = gw.simulate(probs, n_gen, seed=r * 2749 + 1, max_pop=5000)
        trajs.append(sizes)
        max_size = max(max_size, max(sizes))

    import math
    def ly(pop):
        # log scale (pop 0 -> bottom)
        if pop <= 0:
            return ly1
        return ly1 - math.log(pop + 1) / math.log(max_size + 1) * (ly1 - ly0)

    for sizes in trajs:
        extinct = sizes[-1] == 0
        col = P["red"] if extinct else P["green"]
        pts = " ".join(f"{lx(g):.1f},{ly(sizes[g]):.1f}" for g in range(len(sizes)))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="0.9" opacity="0.6"/>')
    parts.append(f'<rect x="{lx0}" y="{ly0}" width="{lx1-lx0}" height="{ly1-ly0}" fill="none" '
                 f'stroke="{P["gray"]}" stroke-width="1"/>')
    parts.append(f'<text x="{lx0}" y="{ly0 - 4:.1f}" fill="{P["gray"]}" font-size="10">'
                 f'lineage size (log) vs generation</text>')
    parts.append(f'<text x="{lx0}" y="{ly1 + 14:.1f}" fill="{P["red"]}" font-size="10">'
                 f'red = extinct</text>')
    parts.append(f'<text x="{lx0 + 90}" y="{ly1 + 14:.1f}" fill="{P["green"]}" font-size="10">'
                 f'green = surviving/growing</text>')

    parts.append(f'<text x="20" y="{H - 24}" fill="{P["gray"]}" font-size="11">'
                 f'iterating q=G(q) from 0 (green cobweb) climbs to the extinction probability -- the '
                 f'smaller of the two fixed points.</text>')
    parts.append(f'<text x="20" y="{H - 8}" fill="{P["gray"]}" font-size="11">'
                 f'about 40% of these supercritical lineages die out; the rest grow geometrically.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
