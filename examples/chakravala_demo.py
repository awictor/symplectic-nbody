"""Chakravala demo: solve Pell's equation by Bhaskara II's cyclic method, tracing the triples (SVG)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import chakravala as CH


BG = "#0d1117"
TEXT = "#e6edf3"
GRAY = "#8b949e"
BLUE = "#4dabf7"
GREEN = "#06d6a0"
YELLOW = "#ffd43b"
RED = "#ff6b6b"


def main(outdir=None):
    lines = []
    lines.append("Chakravala: Bhaskara II's cyclic method for x^2 - D y^2 = 1")
    lines.append("=" * 60)
    lines.append("engine: Brahmagupta's identity (bhavana) -- compose triples, divide by k,")
    lines.append("pick the auxiliary m minimizing |m^2 - D|. All exact integer arithmetic.")
    lines.append("")

    # trace the hard case D = 61
    D = 61
    x, y, triples = CH.fundamental_solution(D, trace=True)
    lines.append(f"D = {D} (a famously hard case; smallest solution found in {len(triples)-1} steps):")
    lines.append(f"{'step':>5}{'a':>16}{'b':>14}{'k':>8}")
    for i, (a, b, k) in enumerate(triples):
        lines.append(f"{i:>5}{a:>16}{b:>14}{k:>8}")
    lines.append("")
    lines.append(f"fundamental solution: x = {x}, y = {y}")
    lines.append(f"check x^2 - {D} y^2 = {x*x - D*y*y} (== 1)")
    lines.append("")
    # a few more D
    lines.append("fundamental solutions for a range of D:")
    lines.append(f"{'D':>5}{'x':>18}{'y':>16}{'steps':>8}")
    for Dv in (2, 13, 29, 46, 61, 94, 109):
        xx, yy, tr = CH.fundamental_solution(Dv, trace=True)
        lines.append(f"{Dv:>5}{xx:>18}{yy:>16}{len(tr)-1:>8}")
    lines.append("")
    lines.append("Chakravala predates the European (continued-fraction) solution by ~600 years")
    lines.append("and reaches even D=61's ten-digit answer in only a handful of cyclic steps.")

    text = "\n".join(lines)
    print(text)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        # SVG: the |k| values over the cycle, showing it homing in on k=1
        W, H = 680, 360
        ml, mt, w, h = 60, 55, 580, 240
        ks = [abs(k) for _, _, k in triples]
        n = len(ks)
        kmax = max(ks) or 1

        def sx(i):
            return ml + (i / (n - 1) if n > 1 else 0) * w

        def sy(v):
            return mt + h - v / kmax * h

        s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
        s.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
        s.append(f'<text x="{ml}" y="30" fill="{TEXT}" font-size="15">'
                 f'Chakravala on D={D}: the auxiliary |k| cycles down to 1</text>')
        s.append(f'<line x1="{ml}" y1="{mt+h}" x2="{ml+w}" y2="{mt+h}" stroke="{GRAY}"/>')
        s.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+h}" stroke="{GRAY}"/>')
        # k=1 target line
        s.append(f'<line x1="{ml}" y1="{sy(1):.1f}" x2="{ml+w}" y2="{sy(1):.1f}" '
                 f'stroke="{GREEN}" stroke-width="1" stroke-dasharray="4,3"/>')
        s.append(f'<text x="{ml+w-40}" y="{sy(1)-4:.1f}" fill="{GREEN}" font-size="10">k=1</text>')
        # bars
        for i, kv in enumerate(ks):
            col = GREEN if kv == 1 else (RED if kv > kmax * 0.5 else YELLOW)
            bx = sx(i)
            s.append(f'<circle cx="{bx:.1f}" cy="{sy(kv):.1f}" r="4" fill="{col}"/>')
            if i > 0:
                s.append(f'<line x1="{sx(i-1):.1f}" y1="{sy(ks[i-1]):.1f}" x2="{bx:.1f}" '
                         f'y2="{sy(kv):.1f}" stroke="{BLUE}" stroke-width="1.5"/>')
            s.append(f'<text x="{bx:.1f}" y="{sy(kv)-8:.1f}" fill="{TEXT}" font-size="9" '
                     f'text-anchor="middle">{ks[i] if kv!=abs(0) else ""}</text>')
        for i in range(n):
            s.append(f'<text x="{sx(i):.1f}" y="{mt+h+16}" fill="{GRAY}" font-size="9" '
                     f'text-anchor="middle">{i}</text>')
        s.append(f'<text x="{ml+w/2:.0f}" y="{mt+h+34}" fill="{GRAY}" font-size="11" '
                 f'text-anchor="middle">cyclic step</text>')
        s.append(f'<text x="{ml}" y="{H-12}" fill="{GRAY}" font-size="10">'
                 f'Each step composes the current triple with an auxiliary chosen to shrink |k|; '
                 f'when |k| hits 1 the triple (a,b) IS the fundamental solution.</text>')
        s.append("</svg>")
        with open(os.path.join(outdir, "chakravala.svg"), "w", encoding="utf-8") as fh:
            fh.write("".join(s))

    return text


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
