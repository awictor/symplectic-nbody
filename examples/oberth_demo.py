"""Demo: the Oberth effect -- burn low and fast for free energy.

Shows that the same burn dv yields far more energy (and hyperbolic excess speed)
when made deep in the gravity well at high speed. Renders the resulting
v_infinity vs burn radius on a fixed eccentric orbit to SVG.

    python examples/oberth_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from oberth import (energy_gain, speed_at_radius, v_infinity_after_burn,  # noqa: E402
                    oberth_advantage, MU_EARTH, R_EARTH)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    rp, ra = R_EARTH + 300e3, R_EARTH + 35786e3
    a = 0.5 * (rp + ra)
    dv = 1500.0

    print("The Oberth effect: the same burn buys more energy at high speed\n")
    print(f"  same {dv:.0f} m/s burn on a 300 km x 35786 km orbit:")
    v_peri = speed_at_radius(MU_EARTH, rp, a)
    v_apo = speed_at_radius(MU_EARTH, ra, a)
    print(f"    at periapsis (v={v_peri/1e3:.1f} km/s): v_inf = "
          f"{v_infinity_after_burn(MU_EARTH, rp, v_peri, dv)/1e3:.2f} km/s (escapes)")
    print(f"    at apoapsis  (v={v_apo/1e3:.1f} km/s): v_inf = "
          f"{v_infinity_after_burn(MU_EARTH, ra, v_apo, dv)/1e3:.2f} km/s (still bound)")
    print(f"    energy-gain advantage of the periapsis burn: "
          f"{oberth_advantage(MU_EARTH, rp, ra, dv, a):.1f}x\n")
    print("  This is why probes dive toward a planet before their escape burn, and")
    print("  why a powered gravity assist (an Oberth maneuver at closest approach)")
    print("  extracts far more than the same burn in deep space.")

    _svg(rp, ra, a, dv, os.path.join(outdir, "oberth.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'oberth.svg')}")


def _svg(rp, ra, a, dv, path, size=720, pad=64):
    # v_infinity vs burn radius along the orbit
    rs = [rp + (ra - rp) * i / 200 for i in range(201)]
    vinf = [v_infinity_after_burn(MU_EARTH, r, speed_at_radius(MU_EARTH, r, a), dv) / 1e3
            for r in rs]
    rmin_km, rmax_km = rp / 1e3, ra / 1e3
    vmax = max(vinf) * 1.1

    def sx(r_km):
        return pad + (r_km - rmin_km) / (rmax_km - rmin_km) * (size - 2 * pad)

    def sy(v):
        return size - pad - v / vmax * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    poly = " ".join(f"{sx(rs[i]/1e3):.1f},{sy(vinf[i]):.1f}" for i in range(len(rs)))
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#4cc9f0" stroke-width="2.2"/>')
    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Oberth effect: escape speed from a fixed burn vs burn radius</text>')
    parts.append(f'<text x="{pad}" y="{size-pad+24}" fill="#8b949e" font-size="11">'
                 f'burn radius (km) -- lower/faster = more v_infinity for the same dv</text>')
    parts.append(f'<text x="{pad-10}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'hyperbolic excess speed (km/s)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
