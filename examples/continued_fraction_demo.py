"""Demo: continued fractions and the best rational approximations of famous constants.

Expands pi, e, the golden ratio, and sqrt(2) as continued fractions, shows their convergents closing
in geometrically, and draws the convergent error falling with denominator. Highlights why 355/113 is
such a good approximation of pi.

    python examples/continued_fraction_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from continued_fraction import cf_expansion, convergents, approximation_error  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Continued fractions: the best rational approximations of a real\n")

    consts = [("pi", math.pi), ("e", math.e), ("golden ratio", (1 + math.sqrt(5)) / 2),
              ("sqrt(2)", math.sqrt(2))]
    for name, x in consts:
        terms = cf_expansion(x, 10)
        print(f"  {name:14} = [{terms[0]}; {', '.join(map(str, terms[1:8]))}, ...]")

    print("\n  Convergents of pi (each the best rational for its denominator):")
    c = convergents(cf_expansion(math.pi, 8))
    for f in c[:6]:
        err = approximation_error(math.pi, f)
        print(f"    {str(f):>16}  = {float(f):.10f}   error {err:.2e}")
    print(f"    355/113 is accurate to 7 digits because pi's next quotient (292) is huge --")
    print(f"    a large partial quotient means the preceding convergent is exceptionally good.")

    # error vs denominator for several constants
    print("\n  Convergent error falls roughly as 1/denominator^2:")
    for name, x in [("pi", math.pi), ("e", math.e)]:
        c = convergents(cf_expansion(x, 8))
        line = "  ".join(f"{f.denominator}:{approximation_error(x, f):.1e}" for f in c[1:6])
        print(f"    {name}: {line}")

    _svg(os.path.join(outdir, "continued_fraction.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'continued_fraction.svg')}")


def _svg(path, width=760, height=430):
    m_left, m_bot, m_top, m_right = 70, 60, 80, 130
    pw = width - m_left - m_right
    ph = height - m_top - m_bot

    series = [("pi", math.pi, "#4dabf7"), ("e", math.e, "#06d6a0"),
              ("golden ratio", (1 + math.sqrt(5)) / 2, "#ffd43b"), ("sqrt(2)", math.sqrt(2), "#ff6b6b")]

    # collect (log10 denominator, log10 error) for each convergent
    all_pts = []
    for name, x, col in series:
        c = convergents(cf_expansion(x, 10))
        pts = []
        for f in c[1:]:
            err = approximation_error(x, f)
            if err > 0 and f.denominator > 1:
                pts.append((math.log10(f.denominator), math.log10(err)))
        all_pts.append((name, col, pts))

    xs = [p[0] for _, _, pts in all_pts for p in pts]
    ys = [p[1] for _, _, pts in all_pts for p in pts]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    def px(lx):
        return m_left + (lx - xmin) / (xmax - xmin) * pw

    def py(ly):
        return m_top + ph - (ly - ymin) / (ymax - ymin) * ph

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">'
        f'Continued-fraction convergents: approximation error vs denominator</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'log-log: error falls steeply with denominator; the golden ratio (worst-approximable) falls slowest</text>',
    ]
    parts.append(f'<line x1="{m_left}" y1="{m_top+ph}" x2="{m_left+pw}" y2="{m_top+ph}" '
                 f'stroke="#484f58" stroke-width="1.5"/>')
    parts.append(f'<line x1="{m_left}" y1="{m_top}" x2="{m_left}" y2="{m_top+ph}" '
                 f'stroke="#484f58" stroke-width="1.5"/>')
    parts.append(f'<text x="{m_left+pw/2:.0f}" y="{height-14}" fill="#8b949e" font-size="12" '
                 f'text-anchor="middle">log10(denominator)</text>')

    ly = m_top + 6
    for name, col, pts in all_pts:
        line = " ".join(f"{px(a):.1f},{py(b):.1f}" for a, b in pts)
        parts.append(f'<polyline points="{line}" fill="none" stroke="{col}" stroke-width="1.8"/>')
        for a, b in pts:
            parts.append(f'<circle cx="{px(a):.1f}" cy="{py(b):.1f}" r="3" fill="{col}"/>')
        parts.append(f'<line x1="{m_left+pw+12}" y1="{ly}" x2="{m_left+pw+30}" y2="{ly}" '
                     f'stroke="{col}" stroke-width="2.5"/>')
        parts.append(f'<text x="{m_left+pw+34}" y="{ly+4}" fill="#e6edf3" font-size="10">{name}</text>')
        ly += 16

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
