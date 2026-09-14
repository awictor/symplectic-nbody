"""Catmull-Rom demo: uniform vs centripetal vs chordal through the same points (SVG showing overshoot)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import catmull_rom as CR


BG = "#0d1117"
TEXT = "#e6edf3"
GRAY = "#8b949e"
YELLOW = "#ffd43b"
RED = "#ff6b6b"
GREEN = "#06d6a0"
BLUE = "#4dabf7"


def _svg(path, pts):
    W, H = 720, 420
    ml, mt, w, h = 50, 50, 620, 300
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    lo_x, hi_x = min(xs) - 1, max(xs) + 1
    lo_y, hi_y = min(ys) - 2, max(ys) + 2

    def sx(x):
        return ml + (x - lo_x) / (hi_x - lo_x) * w

    def sy(y):
        return mt + h - (y - lo_y) / (hi_y - lo_y) * h

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" font-family="monospace">']
    s.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
    s.append(f'<text x="{ml}" y="28" fill="{TEXT}" font-size="15">'
             f'Catmull-Rom parameterizations through the same points</text>')
    # control polygon
    poly = " ".join(f"{sx(p[0]):.1f},{sy(p[1]):.1f}" for p in pts)
    s.append(f'<polyline points="{poly}" fill="none" stroke="{GRAY}" '
             f'stroke-width="0.8" stroke-dasharray="3,3"/>')
    # three curves
    for kind, col in (("uniform", RED), ("chordal", BLUE), ("centripetal", GREEN)):
        c = CR.CatmullRom(pts, kind)
        sp = c.sample(60)
        cv = " ".join(f"{sx(p[0]):.1f},{sy(p[1]):.1f}" for p in sp)
        s.append(f'<polyline points="{cv}" fill="none" stroke="{col}" stroke-width="2"/>')
    # control points
    for p in pts:
        s.append(f'<circle cx="{sx(p[0]):.1f}" cy="{sy(p[1]):.1f}" r="4" fill="{YELLOW}"/>')
    # legend
    ly = mt + 20
    for kind, col in (("uniform (overshoots)", RED), ("chordal", BLUE),
                      ("centripetal (no cusp/overshoot)", GREEN)):
        s.append(f'<line x1="{ml+w-210}" y1="{ly}" x2="{ml+w-190}" y2="{ly}" '
                 f'stroke="{col}" stroke-width="2.5"/>')
        s.append(f'<text x="{ml+w-184}" y="{ly+4}" fill="{TEXT}" font-size="11">{kind}</text>')
        ly += 20
    s.append(f'<text x="{ml}" y="{H-16}" fill="{GRAY}" font-size="10">'
             f'All three interpolate the yellow points, but uniform spacing overshoots and loops near '
             f'the tight corner; centripetal (alpha=0.5) is provably free of cusps and self-crossings.</text>')
    s.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("".join(s))


def _dist(a, b):
    return sum((a[i] - b[i]) ** 2 for i in range(len(a))) ** 0.5


def main(outdir=None):
    # wide-then-tight spacing near a sharp corner: the classic overshoot demo
    pts = [[0, 0], [10, 0], [11, 5], [12, 0], [22, 0]]

    lines = []
    lines.append("Catmull-Rom interpolating splines: three parameterizations")
    lines.append("=" * 60)
    lines.append(f"control points: {pts}")
    lines.append("knot spacing t_{i+1} - t_i = |P_{i+1} - P_i|^alpha")
    lines.append("  alpha = 0 uniform,  0.5 centripetal,  1 chordal")
    lines.append("")

    # overshoot measurement on the sharp segment (index 1: from P1 to P2)
    lines.append("Overshoot on the tight corner segment (P1->P2, x should stay in [10, 11]):")
    lines.append(f"{'kind':>13}{'x-min':>9}{'x-max':>9}{'overshoot':>11}")
    for kind in ("uniform", "centripetal", "chordal"):
        c = CR.CatmullRom(pts, kind)
        xs = [c.evaluate_segment(1, k / 60)[0] for k in range(61)]
        over = max(0.0, 10 - min(xs)) + max(0.0, max(xs) - 11)
        lines.append(f"{kind:>13}{min(xs):>9.3f}{max(xs):>9.3f}{over:>11.4f}")
    lines.append("Uniform overshoots the corner (can loop); centripetal/chordal stay put.")
    lines.append("")

    # arc length of each curve
    lines.append("Total sampled arc length (uniform inflates near uneven spacing):")
    for kind in ("uniform", "centripetal", "chordal"):
        c = CR.CatmullRom(pts, kind)
        sp = c.sample(80)
        L = sum(_dist(sp[i], sp[i + 1]) for i in range(len(sp) - 1))
        lines.append(f"  {kind:>13}: {L:.3f}")
    lines.append("")
    lines.append("All three pass exactly through every control point (interpolating, not approximating).")

    text = "\n".join(lines)
    print(text)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        _svg(os.path.join(outdir, "catmull_rom.svg"), pts)

    return text


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
