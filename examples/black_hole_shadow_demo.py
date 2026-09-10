"""Demo: the black-hole shadow imaged by the Event Horizon Telescope.

Prints the predicted shadow angular size for the EHT targets, then draws the nested
radii to scale -- event horizon, photon sphere, and the lensed shadow edge -- showing
why the dark disk is ~5.2 Schwarzschild radii across, larger than the horizon.

    python examples/black_hole_shadow_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from black_hole_shadow import (schwarzschild_radius, photon_sphere_radius,  # noqa: E402
                               critical_impact_parameter, shadow_angular_diameter_uas,
                               shadow_in_rs, M_SUN, PC, MPC)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Black-hole shadow: apparent size 3 sqrt(3) r_s = 5.2 r_s (EHT)\n")
    print(f"  {'object':>10}{'mass (Msun)':>14}{'distance':>12}"
          f"{'shadow (uas)':>14}")
    print("  " + "-" * 50)
    targets = [
        ("M87*", 6.5e9, 16.8 * MPC, "16.8 Mpc"),
        ("Sgr A*", 4.15e6, 8150 * PC, "8.15 kpc"),
        ("stellar BH", 10.0, 3000 * PC, "3 kpc"),
    ]
    for name, m_sun, D, dstr in targets:
        theta = shadow_angular_diameter_uas(m_sun * M_SUN, D)
        theta_str = f"{theta:.1f}" if theta > 1e-3 else f"{theta:.1e}"
        print(f"  {name:>10}{m_sun:>14.2e}{dstr:>12}{theta_str:>14}")

    print(f"\n  shadow diameter = {shadow_in_rs(M_SUN):.3f} r_s (vs 2 r_s for the horizon)")
    print("\n  Light is captured inside the critical impact parameter b = 3 sqrt(3) GM/c^2,")
    print("  so the dark disk is bigger than the event horizon -- gravitational lensing")
    print("  magnifies it. M87* and Sgr A* both subtend only ~40-50 microarcseconds, the")
    print("  angular size of an orange on the Moon, which is why the EHT had to link radio")
    print("  dishes across the whole Earth to resolve them.")

    _svg(os.path.join(outdir, "black_hole_shadow.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'black_hole_shadow.svg')}")


def _svg(path, size=640):
    cx = cy = size / 2.0
    M = M_SUN
    rs = schwarzschild_radius(M)
    # scale so the shadow (5.196 r_s radius... diameter) fits nicely
    scale = (size * 0.38) / (shadow_in_rs(M) / 2.0)   # px per r_s

    r_horizon = 1.0 * scale          # horizon radius = 1 r_s
    r_photon = 1.5 * scale           # photon sphere = 1.5 r_s
    r_shadow = shadow_in_rs(M) / 2.0 * scale   # shadow radius = 2.598 r_s

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
    ]
    # glowing accretion ring just outside the shadow
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r_shadow*1.06:.1f}" fill="none" '
                 f'stroke="#ff922b" stroke-width="10" opacity="0.55"/>')
    # shadow (dark disk)
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r_shadow:.1f}" fill="#000000" '
                 f'stroke="#ffd43b" stroke-width="2"/>')
    # photon sphere
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r_photon:.1f}" fill="none" '
                 f'stroke="#4dabf7" stroke-width="1.4" stroke-dasharray="4 4"/>')
    # event horizon
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r_horizon:.1f}" fill="none" '
                 f'stroke="#ff6b6b" stroke-width="1.4" stroke-dasharray="3 3"/>')

    parts.append(f'<text x="{cx}" y="{cy+r_shadow+26:.1f}" fill="#ffd43b" '
                 f'font-size="12" text-anchor="middle">shadow edge (2.6 r_s)</text>')
    parts.append(f'<text x="{cx}" y="{cy-r_photon-8:.1f}" fill="#4dabf7" '
                 f'font-size="11" text-anchor="middle">photon sphere (1.5 r_s)</text>')
    parts.append(f'<text x="{cx}" y="{cy+4:.1f}" fill="#ff6b6b" '
                 f'font-size="10" text-anchor="middle">horizon 1 r_s</text>')

    parts.append(f'<text x="20" y="34" fill="#e6edf3" font-size="18">'
                 f'The black-hole shadow (to scale)</text>')
    parts.append(f'<text x="20" y="52" fill="#8b949e" font-size="12">'
                 f'lensing makes the dark disk 5.2 r_s across -- bigger than the horizon</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
