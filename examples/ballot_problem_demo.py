"""Ballot-problem demo: Bertrand's (p-q)/(p+q) law, the reflection principle, and lattice paths staying above 0."""

import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import ballot_problem as bp


PALETTE = {
    "bg": "#0d1117", "blue": "#4dabf7", "yellow": "#ffd43b", "red": "#ff6b6b",
    "green": "#06d6a0", "purple": "#b197fc", "gray": "#8b949e", "text": "#e6edf3",
}


def main():
    lines = []
    lines.append("Bertrand's ballot problem -- P(candidate A leads the entire count) = (p-q)/(p+q)")
    lines.append("=" * 78)
    lines.append("")
    lines.append("A finishes with p votes, B with q < p. Counted in random order, how often is A")
    lines.append("strictly ahead at every step? Depends only on the margin relative to turnout:")
    lines.append("")
    lines.append("     p     q    margin    P(lead throughout)   good orderings / all")
    lines.append("   " + "-" * 62)
    import math
    for p, q in [(3, 2), (5, 3), (10, 8), (6, 1), (8, 2), (10, 0)]:
        prob = bp.ballot_probability(p, q)
        good = bp.ballot_count_strict(p, q)
        total = math.comb(p + q, p)
        lines.append(f"   {p:4d}  {q:4d}   {p-q:5d}       {float(prob):7.4f}          {good:6d} / {total:<6d}")
    lines.append("")
    lines.append("A landslide is almost never behind; a dead heat (p=q+1) leads throughout only 1/(p+q).")
    lines.append("")

    lines.append("Three independent derivations of the strict count agree exactly:")
    lines.append("   (p,q)    formula   reflection   cycle-lemma   brute-force")
    lines.append("   " + "-" * 56)
    for p, q in [(4, 2), (5, 3), (6, 2), (7, 4)]:
        f = bp.ballot_count_strict(p, q)
        r = bp.ballot_count_strict_reflection(p, q)
        c = bp.ballot_count_cycle_lemma(p, q)
        b = bp.brute_force_strict(p, q)
        lines.append(f"   ({p},{q})     {f:5d}     {r:5d}       {c:5d}         {b:5d}")
    lines.append("")

    lines.append("At p = q (weak lead, ties allowed) the count IS the Catalan numbers -- Dyck paths:")
    cats = ", ".join(str(bp.catalan(n)) for n in range(9))
    lines.append(f"   catalan(0..8) = {cats}")

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
                 f'Vote count as a lattice path: +1 per A vote, -1 per B vote</text>')

    # geometry
    p, q = 5, 3
    steps = p + q
    x0, x1 = 60, W - 40
    axis_y = 250          # height 0
    unit_x = (x1 - x0) / steps
    unit_y = 26

    def px(i):
        return x0 + i * unit_x

    def py(h):
        return axis_y - h * unit_y

    # gridlines for heights
    for h in range(-3, 6):
        y = py(h)
        col = P["text"] if h == 0 else P["gray"]
        w = 1.2 if h == 0 else 0.4
        parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="{col}" '
                     f'stroke-width="{w}" opacity="{0.9 if h==0 else 0.35}"/>')
        parts.append(f'<text x="{x0-8}" y="{y+4:.1f}" fill="{P["gray"]}" font-size="9" text-anchor="end">{h:+d}</text>')

    def draw_path(seq, col, label, ly, dash=""):
        h = 0
        pts = [(px(0), py(0))]
        for i, s in enumerate(seq):
            h += 1 if s == "A" else -1
            pts.append((px(i + 1), py(h)))
        pstr = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        d = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<polyline points="{pstr}" fill="none" stroke="{col}" stroke-width="2"{d}/>')
        for x, y in pts:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.5" fill="{col}"/>')
        parts.append(f'<text x="{x1-4:.1f}" y="{ly}" fill="{col}" font-size="11" text-anchor="end">{label}</text>')

    # a GOOD path (stays strictly above 0 after step 1) and a BAD path (touches 0)
    draw_path("AABABAB", P["green"], "GOOD: A leads throughout (stays > 0)", 60)
    draw_path("ABBAABA", P["red"], "BAD: touches 0 -- lead is lost", 78, dash="5,4")

    parts.append(f'<text x="20" y="{H-42}" fill="{P["gray"]}" font-size="11">'
                 f'The reflection principle counts BAD paths by reflecting their first-return segment across</text>')
    parts.append(f'<text x="20" y="{H-26}" fill="{P["gray"]}" font-size="11">'
                 f'the axis, biject-ing them with paths that start downward. Good paths = all minus reflected,</text>')
    parts.append(f'<text x="20" y="{H-10}" fill="{P["gray"]}" font-size="11">'
                 f'which collapses to the fraction (p-q)/(p+q) of every ordering.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
