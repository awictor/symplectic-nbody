"""Demo: the Hohmann transfer -- mission delta-v budgets.

Computes the two-burn transfer for LEO->GEO and Earth->Mars, prints the delta-v
budget and trip time, and renders the transfer geometry (inner orbit, transfer
ellipse, outer orbit) to SVG.

    python examples/hohmann_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hohmann import (transfer, phase_angle, rocket_equation_mass_ratio,  # noqa: E402
                     MU_EARTH, MU_SUN, AU, DAY)

RE = 6.378e6


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Hohmann transfer: the cheapest way between two circular orbits\n")

    r1, r2 = RE + 200e3, RE + 35786e3
    dv1, dv2, dvt, t = transfer(r1, r2, MU_EARTH)
    print("  LEO (200 km) -> GEO (35786 km):")
    print(f"    burn 1 = {dv1:.0f} m/s, burn 2 = {dv2:.0f} m/s, total = {dvt:.0f} m/s")
    print(f"    transfer time = {t/3600:.2f} hours")
    print(f"    propellant: m0/mf = {rocket_equation_mass_ratio(dvt, 4400):.2f} "
          f"(LH2/LOX, ve=4.4 km/s)\n")

    dv1, dv2, dvt, t = transfer(1.0 * AU, 1.524 * AU, MU_SUN)
    print("  Earth -> Mars (heliocentric):")
    print(f"    total delta-v = {dvt/1e3:.2f} km/s")
    print(f"    transfer time = {t/DAY:.0f} days")
    print(f"    launch phase angle = {math.degrees(phase_angle(1.0*AU, 1.524*AU, MU_SUN)):.1f} deg\n")
    print("  These delta-v budgets set the propellant mass via the rocket equation,")
    print("  and the phase angle sets the launch window -- why Mars missions leave")
    print("  Earth only every ~26 months.")

    _svg(1.0, 1.524, os.path.join(outdir, "hohmann.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'hohmann.svg')}")


def _svg(r1_au, r2_au, path, size=680):
    cx = cy = size // 2
    scale = (size // 2 - 40) / r2_au

    def px(x): return cx + x * scale
    def py(y): return cy - y * scale

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#05070d"/>',
    ]
    # Sun
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="6" fill="#ffd166"/>')
    # inner (Earth) and outer (Mars) circular orbits
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r1_au*scale:.1f}" fill="none" '
                 f'stroke="#4cc9f0" stroke-width="1.5"/>')
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r2_au*scale:.1f}" fill="none" '
                 f'stroke="#ff006e" stroke-width="1.5"/>')
    # transfer ellipse: periapsis at r1 (+x), apoapsis at r2 (-x)
    a = 0.5 * (r1_au + r2_au)
    b = math.sqrt(r1_au * r2_au)     # semi-minor of the transfer ellipse
    cxe = cx - (a - r1_au) * scale   # ellipse center shifted toward apoapsis (-x)
    parts.append(f'<ellipse cx="{cxe:.1f}" cy="{cy}" rx="{a*scale:.1f}" '
                 f'ry="{b*scale:.1f}" fill="none" stroke="#e9c46a" '
                 f'stroke-width="1.8" stroke-dasharray="6,4"/>')
    # departure (periapsis, +x) and arrival (apoapsis, -x)
    parts.append(f'<circle cx="{px(r1_au):.1f}" cy="{py(0):.1f}" r="5" fill="#4cc9f0"/>')
    parts.append(f'<circle cx="{px(-r2_au):.1f}" cy="{py(0):.1f}" r="5" fill="#ff006e"/>')
    parts.append(f'<text x="16" y="26" fill="#e6edf3" font-size="16">'
                 f'Hohmann transfer: Earth (blue) -&gt; Mars (pink)</text>')
    parts.append(f'<text x="16" y="{size-16}" fill="#8b949e" font-size="11">'
                 f'gold dashed = transfer ellipse; burns at periapsis and apoapsis</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
