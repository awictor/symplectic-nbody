"""Demo: the Hall effect -- weighing charge carriers with a magnet.

Prints the Hall voltage, coefficient, mobility and carrier sign across materials from copper
to a semiconductor, then draws a Hall bar: current along it, field through it, carriers
deflected to one edge, and the transverse Hall voltage that results.

    python examples/hall_effect_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hall_effect import (hall_voltage, hall_coefficient, hall_mobility,  # noqa: E402
                         hall_angle, carrier_sign, E_CHARGE)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Hall effect: V_H = I B / (n q t) reveals carrier density AND sign\n")
    print("  1 mA through a 0.1 mm-thick sample in a 1 T field:\n")
    print(f"  {'material':<22}{'n (1/m^3)':>12}{'carrier':>11}{'V_H':>12}{'|R_H|':>12}")
    # (name, n, charge sign, sigma)
    materials = [
        ("copper (electrons)", 8.5e28, -1, 6e7),
        ("aluminium (holes*)", 1.8e29, +1, 3.8e7),
        ("n-Si (doped)", 1e22, -1, 1e3),
        ("p-Ge (doped)", 5e21, +1, 200.0),
    ]
    for name, n, sign, sigma in materials:
        q = sign * E_CHARGE
        V = hall_voltage(1e-3, 1.0, n, 1e-4, charge=q)
        RH = hall_coefficient(n, charge=q)
        av = abs(V)
        vs = f"{av*1e6:.3f} uV" if av < 1e-3 else (f"{av*1e3:.2f} mV" if av < 1.0 else f"{av:.2f} V")
        print(f"  {name:<22}{n:>12.1e}{carrier_sign(RH):>11}{vs:>12}{abs(RH):>12.2e}")

    print("\n  (*Al's measured Hall sign is positive -- a famous hint that band structure makes")
    print("  some carriers behave as positive 'holes', which classical free electrons can't explain.)")
    print("\n  The doped semiconductors give millivolt Hall signals from their sparse carriers,")
    print("  vs microvolts in a metal: fewer carriers, bigger V_H. Mobility follows from")
    print("  mu = |R_H| sigma -- n-Si here: mu = %.3f m^2/Vs, Hall angle %.1f deg at 1 T."
          % (hall_mobility(abs(hall_coefficient(1e22)), 1e3),
             math.degrees(hall_angle(hall_mobility(abs(hall_coefficient(1e22)), 1e3), 1.0))))

    _svg(os.path.join(outdir, "hall_effect.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'hall_effect.svg')}")


def _svg(path, size=720, pad=70):
    # Hall bar: rectangular slab, current arrow along x, B dots out of page, carriers
    # deflected to the bottom edge, + / - on the two transverse edges.
    x0, x1 = pad, size - pad
    cy = size * 0.44
    bar_w = x1 - x0
    bar_h = size * 0.26
    top = cy - bar_h / 2
    bot = cy + bar_h / 2

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="34" fill="#e6edf3" font-size="18">The Hall bar</text>',
        f'<text x="20" y="52" fill="#8b949e" font-size="12">'
        f'current + perpendicular field push carriers sideways, building a transverse Hall voltage</text>',
    ]

    # the conductor slab
    parts.append(f'<rect x="{x0:.1f}" y="{top:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" '
                 f'fill="#30363d" stroke="#8b949e" stroke-width="2"/>')

    # field out of page (dots)
    for fx in range(6):
        for fy in range(2):
            dx = x0 + bar_w * (0.15 + 0.14 * fx)
            dy = top + bar_h * (0.35 + 0.35 * fy)
            parts.append(f'<circle cx="{dx:.1f}" cy="{dy:.1f}" r="2.5" fill="none" '
                         f'stroke="#b197fc" stroke-width="1.4"/>')
            parts.append(f'<circle cx="{dx:.1f}" cy="{dy:.1f}" r="0.8" fill="#b197fc"/>')
    parts.append(f'<text x="{x0+bar_w*0.5:.1f}" y="{top-8:.1f}" fill="#b197fc" font-size="11" '
                 f'text-anchor="middle">B out of page</text>')

    # current arrow (conventional current +x)
    ay = cy
    parts.append(f'<line x1="{x0-30:.1f}" y1="{ay:.1f}" x2="{x0:.1f}" y2="{ay:.1f}" '
                 f'stroke="#ffd43b" stroke-width="3"/>')
    parts.append(f'<line x1="{x1:.1f}" y1="{ay:.1f}" x2="{x1+30:.1f}" y2="{ay:.1f}" '
                 f'stroke="#ffd43b" stroke-width="3"/>')
    parts.append(f'<polygon points="{x1+30:.1f},{ay:.1f} {x1+18:.1f},{ay-6:.1f} '
                 f'{x1+18:.1f},{ay+6:.1f}" fill="#ffd43b"/>')
    parts.append(f'<text x="{x1+34:.1f}" y="{ay-8:.1f}" fill="#ffd43b" font-size="12">I</text>')

    # deflected carrier drifting to bottom edge
    parts.append(f'<line x1="{x0+bar_w*0.35:.1f}" y1="{cy:.1f}" x2="{x0+bar_w*0.55:.1f}" y2="{bot-8:.1f}" '
                 f'stroke="#4dabf7" stroke-width="1.8"/>')
    parts.append(f'<polygon points="{x0+bar_w*0.55:.1f},{bot-8:.1f} {x0+bar_w*0.53:.1f},{bot-18:.1f} '
                 f'{x0+bar_w*0.585:.1f},{bot-14:.1f}" fill="#4dabf7"/>')
    parts.append(f'<circle cx="{x0+bar_w*0.35:.1f}" cy="{cy:.1f}" r="4" fill="#4dabf7"/>')
    parts.append(f'<text x="{x0+bar_w*0.37:.1f}" y="{cy-8:.1f}" fill="#4dabf7" font-size="11">carrier</text>')
    parts.append(f'<text x="{x0+bar_w*0.5:.1f}" y="{(cy+bot)/2:.1f}" fill="#8b949e" font-size="10">'
                 f'F = q v x B</text>')

    # charge pile-up: - on bottom, + on top (for electrons)
    parts.append(f'<text x="{x0+bar_w*0.5:.1f}" y="{bot+18:.1f}" fill="#ff6b6b" font-size="14" '
                 f'text-anchor="middle">- - - - - -</text>')
    parts.append(f'<text x="{x0+bar_w*0.5:.1f}" y="{top-22:.1f}" fill="#06d6a0" font-size="14" '
                 f'text-anchor="middle">+ + + + + +</text>')

    # Hall voltage probes
    px = x0 + bar_w * 0.8
    parts.append(f'<line x1="{px:.1f}" y1="{top:.1f}" x2="{px:.1f}" y2="{top-40:.1f}" stroke="#8b949e" stroke-width="1.4"/>')
    parts.append(f'<line x1="{px:.1f}" y1="{bot:.1f}" x2="{px:.1f}" y2="{bot+40:.1f}" stroke="#8b949e" stroke-width="1.4"/>')
    parts.append(f'<text x="{px+8:.1f}" y="{cy+4:.1f}" fill="#e6edf3" font-size="13">V_H</text>')

    parts.append(f'<text x="{x0:.1f}" y="{size-40:.1f}" fill="#8b949e" font-size="11">'
                 f'V_H = I B / (n q t): its size gives the carrier density, its sign gives the carrier charge</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
