"""Newton-Cotes demo: degree of exactness per rule, and composite convergence order (SVG)."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from newton_cotes import composite, degree_of_exactness, error_constant, rule_names


BG = "#0d1117"
BLUE = "#4dabf7"
YELLOW = "#ffd43b"
RED = "#ff6b6b"
GREEN = "#06d6a0"
PURPLE = "#b197fc"
ORANGE = "#ff922b"
GRAY = "#8b949e"
TEXT = "#e6edf3"


def _convergence_svg(path):
    # composite error vs panel count (log-log) for trapezoid, simpson, boole on e^x over [0,1]
    ref = math.e - 1
    f = math.exp
    panels = [2, 4, 8, 16, 32, 64, 128]
    series = {
        "trapezoid": (BLUE, 2),
        "simpson": (GREEN, 4),
        "boole": (PURPLE, 6),
    }
    W, H = 640, 380
    ml, mr, mt, mb = 70, 150, 30, 50
    pw, ph = W - ml - mr, H - mt - mb

    # data: log10(panels) vs log10(err)
    data = {}
    all_ly = []
    for rule, (col, order) in series.items():
        pts = []
        for n in panels:
            err = abs(composite(f, 0, 1, rule, n) - ref)
            err = max(err, 1e-17)
            pts.append((math.log10(n), math.log10(err)))
            all_ly.append(math.log10(err))
        data[rule] = pts
    lx = [math.log10(n) for n in panels]
    xmin, xmax = min(lx), max(lx)
    ymin, ymax = min(all_ly), max(all_ly)
    yr = ymax - ymin or 1

    def sx(v):
        return ml + (v - xmin) / (xmax - xmin) * pw

    def sy(v):
        return mt + (ymax - v) / yr * ph

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" font-family="monospace">']
    s.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
    s.append(f'<text x="{ml}" y="18" fill="{TEXT}" font-size="14">Composite error vs panels '
             f'(e^x on [0,1], log-log)</text>')
    # axes
    s.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}" stroke="{GRAY}"/>')
    s.append(f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" stroke="{GRAY}"/>')
    # y gridlines at integer log10
    lo, hi = int(math.floor(ymin)), int(math.ceil(ymax))
    for e in range(lo, hi + 1):
        if e < ymin - 0.001 or e > ymax + 0.001:
            continue
        yy = sy(e)
        s.append(f'<line x1="{ml}" y1="{yy:.1f}" x2="{ml+pw}" y2="{yy:.1f}" '
                 f'stroke="#21262d"/>')
        s.append(f'<text x="{ml-8}" y="{yy+4:.1f}" fill="{GRAY}" font-size="10" '
                 f'text-anchor="end">1e{e}</text>')
    for n in panels:
        xx = sx(math.log10(n))
        s.append(f'<text x="{xx:.1f}" y="{mt+ph+16}" fill="{GRAY}" font-size="10" '
                 f'text-anchor="middle">{n}</text>')
    s.append(f'<text x="{ml+pw/2:.0f}" y="{mt+ph+38}" fill="{GRAY}" font-size="11" '
             f'text-anchor="middle">panels</text>')
    # series
    ly = mt + 20
    for rule, (col, order) in series.items():
        pts = data[rule]
        d = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in pts)
        s.append(f'<polyline points="{d}" fill="none" stroke="{col}" stroke-width="2"/>')
        for x, y in pts:
            s.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="2.5" fill="{col}"/>')
        s.append(f'<rect x="{ml+pw+14}" y="{ly-9}" width="10" height="10" fill="{col}"/>')
        s.append(f'<text x="{ml+pw+28}" y="{ly}" fill="{TEXT}" font-size="11">{rule} '
                 f'O(h^{order})</text>')
        ly += 20
    s.append(f'<text x="{ml+pw+14}" y="{ly+8}" fill="{GRAY}" font-size="10">steeper slope</text>')
    s.append(f'<text x="{ml+pw+14}" y="{ly+22}" fill="{GRAY}" font-size="10">= higher order</text>')
    s.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("".join(s))


def main(outdir=None):
    lines = []
    lines.append("Newton-Cotes closed quadrature rules")
    lines.append("=" * 60)
    lines.append(f"{'rule':<12}{'points':>7}{'deg':>5}{'error term':>14}{'order':>7}")
    npts_map = {"trapezoid": 2, "simpson": 3, "simpson38": 4, "boole": 5, "weddle": 7}
    for rule in rule_names():
        d = degree_of_exactness(rule)
        c, p, k = error_constant(rule)
        lines.append(f"{rule:<12}{npts_map[rule]:>7}{d:>5}{c+' f^('+str(k)+')':>14}{'h^'+str(k):>7}")
    lines.append("")

    # exactness demonstration on [0,1]
    lines.append("Exactness on int_0^1 x^n dx = 1/(n+1):")
    lines.append(f"{'n':>3}{'trapezoid':>12}{'simpson':>12}{'boole':>12}{'exact':>12}")
    for n in range(6):
        row = f"{n:>3}"
        for rule in ("trapezoid", "simpson", "boole"):
            v = composite(lambda x, n=n: x ** n, 0, 1, rule, 1)
            row += f"{v:>12.7f}"
        row += f"{1/(n+1):>12.7f}"
        lines.append(row)
    lines.append("(zeros of error appear once n <= degree of the rule)")
    lines.append("")

    # composite convergence on e^x
    ref = math.e - 1
    lines.append(f"Composite error, int_0^1 e^x dx = e-1 = {ref:.10f}:")
    lines.append(f"{'panels':>8}{'trapezoid':>15}{'simpson':>15}{'boole':>15}")
    for n in (2, 8, 32, 128):
        row = f"{n:>8}"
        for rule in ("trapezoid", "simpson", "boole"):
            err = abs(composite(math.exp, 0, 1, rule, n) - ref)
            row += f"{err:>15.2e}"
        lines.append(row)
    lines.append("Halving h: trapezoid err/4, Simpson err/16, Boole err/64.")

    text = "\n".join(lines)
    print(text)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        _convergence_svg(os.path.join(outdir, "newton_cotes_convergence.svg"))

    return text


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
