"""Demo: solar sails and radiation pressure.

Prints the radiation pressure, sail acceleration and lightness number for real and
proposed sails, then draws the lightness number vs area-to-mass ratio with the beta=1
line (radiation cancels gravity) and the missions marked.

    python examples/solar_sail_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from solar_sail import (solar_flux, radiation_pressure, sail_acceleration,  # noqa: E402
                        lightness_number, critical_area_to_mass, AU)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Solar sail: sunlight pressure 2F/c on a mirror, ~9 uPa at 1 AU\n")
    print(f"  solar flux at 1 AU: {solar_flux(AU):.0f} W/m^2, "
          f"mirror pressure {radiation_pressure(AU)*1e6:.2f} uPa\n")
    print(f"  {'sail':>18}{'area (m^2)':>11}{'mass (kg)':>11}"
          f"{'accel (mm/s^2)':>16}{'beta':>9}")
    print("  " + "-" * 65)
    # (name, area m^2, mass kg)
    sails = [
        ("IKAROS", 196.0, 315.0),
        ("LightSail 2", 32.0, 5.0),
        ("NEA Scout", 86.0, 14.0),
        ("Starshot chip", 16.0, 0.001),
        ("beta = 1 sail", 653.0, 1.0),
    ]
    for name, A, m in sails:
        a = sail_acceleration(A, m, AU) * 1e3
        b = lightness_number(A, m)
        print(f"  {name:>18}{A:>11.0f}{m:>11.3g}{a:>16.4g}{b:>9.3g}")

    print(f"\n  Both sunlight and gravity fall as 1/r^2, so the lightness number beta =")
    print("  radiation force / solar gravity is a fixed property of the sail. beta = 1")
    print("  (area-to-mass ~653 m^2/kg, a ~1.5 g/m^2 mirror) exactly cancels the Sun's")
    print("  pull; beta > 1 escapes the Solar System on sunlight alone. Today's sails")
    print("  sit at beta ~ 0.01 -- gentle, but propellant-free and endless.")

    _svg(os.path.join(outdir, "solar_sail.svg"), sails)
    print(f"\n  wrote {os.path.join(outdir, 'solar_sail.svg')}")


def _svg(path, sails, size=720, pad=72):
    sigmas = [10 ** (-1 + 0.06 * i) for i in range(0, 76)]   # 0.1 .. ~3e3 m^2/kg
    betas = [lightness_number(s, 1.0) for s in sigmas]
    lx = [math.log10(s) for s in sigmas]
    ly = [math.log10(b) for b in betas]
    xmin, xmax = lx[0], lx[-1]
    ymin, ymax = min(ly), max(ly)

    def sx(x):
        return pad + (x - xmin) / (xmax - xmin) * (size - 2 * pad)

    def sy(y):
        return size - pad - (y - ymin) / (ymax - ymin) * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    # beta = 1 line
    if ymin <= 0.0 <= ymax:
        yb = sy(0.0)
        parts.append(f'<line x1="{pad}" y1="{yb:.1f}" x2="{size-pad}" y2="{yb:.1f}" '
                     f'stroke="#ff6b6b" stroke-width="1.3" stroke-dasharray="5 4"/>')
        parts.append(f'<text x="{pad+8}" y="{yb-6:.1f}" fill="#ff6b6b" '
                     f'font-size="11">beta = 1: radiation cancels gravity</text>')

    poly = " ".join(f"{sx(lx[i]):.1f},{sy(ly[i]):.1f}" for i in range(len(sigmas)))
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#ffd43b" stroke-width="2.6"/>')

    for name, A, m in sails:
        sigma = A / m
        b = lightness_number(A, m)
        if xmin <= math.log10(sigma) <= xmax and ymin <= math.log10(b) <= ymax:
            px = sx(math.log10(sigma))
            py = sy(math.log10(b))
            parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.5" fill="#4dabf7"/>')
            parts.append(f'<text x="{px+7:.1f}" y="{py+4:.1f}" fill="#4dabf7" '
                         f'font-size="10">{name}</text>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Solar-sail lightness number vs area-to-mass</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'beta ~ A/m; reach beta = 1 and sunlight balances the Sun\'s gravity</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+24}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">log10 area-to-mass ratio (m^2/kg) -&gt;</text>')
    parts.append(f'<text x="{pad-18}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'log10 lightness number beta</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
