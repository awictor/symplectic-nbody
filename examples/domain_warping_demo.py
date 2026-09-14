"""Domain warping demo: no warp -> 1 level -> 2 levels, the noise folding into swirls (SVG)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import domain_warping as DW
from fbm import total_variation


BG = "#0d1117"
TEXT = "#e6edf3"
GRAY = "#8b949e"


def _color(t):
    """A warm marble-ish palette for the warped field."""
    t = max(0.0, min(1.0, t))
    stops = [(0.0, (20, 20, 45)), (0.35, (60, 50, 110)), (0.6, (200, 110, 90)),
             (0.8, (240, 200, 120)), (1.0, (255, 245, 220))]
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t <= t1:
            w = (t - t0) / (t1 - t0) if t1 > t0 else 0
            return f"#{int(c0[0]+w*(c1[0]-c0[0])):02x}{int(c0[1]+w*(c1[1]-c0[1])):02x}{int(c0[2]+w*(c1[2]-c0[2])):02x}"
    return "#fff5dc"


def main(outdir=None):
    dw = DW.DomainWarp(seed=7, octaves=5, amplitude=1.5)

    lines = []
    lines.append("Domain warping: feeding noise into its own coordinates")
    lines.append("=" * 56)
    lines.append("warp(x) = fbm(x + amplitude * fbm(x + offset)); each level folds the plane again")
    lines.append("")
    lines.append("total variation of a 1-D slice grows with each warp level (more swirl):")
    lines.append(f"{'warp levels':>13}{'total variation':>18}")
    for levels in (0, 1, 2):
        line = [dw.value(x * 0.03, 5.0, levels) for x in range(500)]
        lines.append(f"{levels:>13}{total_variation(line):>18.3f}")
    lines.append("")
    # show the displacement magnitude at a few points
    lines.append("warp displacement vectors (level 1) at sample points:")
    for x, y in [(1.0, 1.0), (3.0, 2.0), (5.5, 4.5)]:
        d = dw.warp_vector(x, y, levels=1)
        lines.append(f"  ({x:.1f}, {y:.1f}) -> displaced by ({d[0]:+.3f}, {d[1]:+.3f})")
    lines.append("")
    lines.append("The inner noise drags the sampling point sideways, turning smooth ridges into")
    lines.append("the whorls and filaments of marble, wood grain, smoke, and magma.")

    text = "\n".join(lines)
    print(text)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        res = 130
        px = 200
        W = 3 * px + 80
        H = px + 110
        s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
        s.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
        s.append(f'<text x="20" y="26" fill="{TEXT}" font-size="15">'
                 f'Domain warping: 0, 1, and 2 levels fold noise into swirls</text>')
        cell = px / res
        for pi, (title, levels) in enumerate([("no warp (plain fBm)", 0),
                                              ("1 warp level", 1),
                                              ("2 warp levels", 2)]):
            ox = 20 + pi * (px + 20)
            oy = 50
            fld = dw.field(res, res, scale=0.045, levels=levels)
            s.append(f'<text x="{ox}" y="{oy-4}" fill="{GRAY}" font-size="11">{title}</text>')
            for j in range(res):
                for i in range(res):
                    s.append(f'<rect x="{ox+i*cell:.2f}" y="{oy+j*cell:.2f}" '
                             f'width="{cell+0.5:.2f}" height="{cell+0.5:.2f}" '
                             f'fill="{_color(fld[j][i])}"/>')
        s.append(f'<text x="20" y="{H-14}" fill="{GRAY}" font-size="10">'
                 f'The same fBm, sampled at coordinates displaced by more fBm; each level of warping '
                 f'adds another fold of organic flow.</text>')
        s.append("</svg>")
        with open(os.path.join(outdir, "domain_warping.svg"), "w", encoding="utf-8") as fh:
            fh.write("".join(s))

    return text


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
