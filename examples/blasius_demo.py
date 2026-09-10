"""Demo: the Blasius boundary layer -- a thin sheared film growing on a flat plate.

Prints the boundary-layer thickness and skin friction along a plate and the laminar-to-
turbulent transition point, then draws the layer growing as sqrt(x) from the leading edge,
with a few velocity profiles rising from zero at the wall to the free stream.

    python examples/blasius_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from blasius import (reynolds_x, bl_thickness, displacement_thickness,  # noqa: E402
                     skin_friction_local, skin_friction_average, drag_force,
                     transition_distance)


U, NU, RHO = 10.0, 1.5e-5, 1.225      # air at 10 m/s


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Blasius boundary layer: delta = 5.0 x / sqrt(Re_x)  (air, U = 10 m/s)\n")
    print(f"  {'x (m)':>8}{'Re_x':>12}{'delta (mm)':>13}{'c_f':>11}")
    for x in (0.01, 0.05, 0.1, 0.3, 0.75):
        print(f"  {x:>8.2f}{reynolds_x(U, x, NU):>12.0f}"
              f"{bl_thickness(U, x, NU)*1000:>13.2f}{skin_friction_local(U, x, NU):>11.5f}")

    xt = transition_distance(U, NU)
    print("\n  Laminar until Re_x ~ 5e5, i.e. x = %.2f m; beyond that it turns turbulent." % xt)

    L = 0.5
    Cd = skin_friction_average(U, L, NU)
    F = drag_force(U, L, 0.2, NU, RHO)
    print("\n  A 0.5 m x 0.2 m plate: C_D = %.5f, friction drag = %.4f N per side." % (Cd, F))
    print("  The layer thickens as sqrt(x) -- a couple of mm over the front of a wing -- while")
    print("  the skin friction thins as 1/sqrt(x), heaviest right at the sharp leading edge.")

    _svg(os.path.join(outdir, "blasius.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'blasius.svg')}")


def _blasius_profile(eta):
    # Approximate normalized Blasius u/U as a function of eta = y/delta (0..1).
    # Use a smooth cubic-ish fit that hits 0 at wall and 1 at edge with the right shape.
    if eta >= 1.0:
        return 1.0
    return 1.5 * eta - 0.5 * eta ** 3        # matches u'(0) and plateau at eta=1


def _svg(path, size=720, pad=70):
    L = 0.9                                  # plate length shown (m)
    x0, x1 = pad, size - pad
    plate_y = size * 0.62
    # exaggerate the thin layer so it is visible
    dmax = bl_thickness(U, L, NU)
    vscale = (size * 0.34) / dmax

    def X(x):
        return x0 + x / L * (x1 - x0)

    def Yd(delta):
        return plate_y - delta * vscale

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="34" fill="#e6edf3" font-size="18">Blasius boundary layer on a flat plate</text>',
        f'<text x="20" y="52" fill="#8b949e" font-size="12">'
        f'the 99%-thickness grows as sqrt(x); no-slip at the wall climbs to the free stream U</text>',
    ]

    # free-stream arrows above the layer
    for yy in (pad + 30, pad + 55):
        parts.append(f'<line x1="{x0-10:.1f}" y1="{yy:.1f}" x2="{x0+40:.1f}" y2="{yy:.1f}" '
                     f'stroke="#8b949e" stroke-width="1.5"/>')
        parts.append(f'<polygon points="{x0+40:.1f},{yy:.1f} {x0+30:.1f},{yy-4:.1f} '
                     f'{x0+30:.1f},{yy+4:.1f}" fill="#8b949e"/>')
    parts.append(f'<text x="{x0+46:.1f}" y="{pad+46:.1f}" fill="#8b949e" font-size="11">free stream U</text>')

    # the plate
    parts.append(f'<line x1="{x0:.1f}" y1="{plate_y:.1f}" x2="{x1:.1f}" y2="{plate_y:.1f}" '
                 f'stroke="#e6edf3" stroke-width="3"/>')

    # delta(x) curve (the boundary-layer edge)
    n = 160
    pts = []
    for i in range(1, n + 1):
        x = L * i / n
        pts.append(f"{X(x):.1f},{Yd(bl_thickness(U, x, NU)):.1f}")
    parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#4dabf7" stroke-width="2.6"/>')
    parts.append(f'<text x="{X(L)-4:.1f}" y="{Yd(bl_thickness(U, L, NU))-6:.1f}" fill="#4dabf7" '
                 f'font-size="11" text-anchor="end">delta(x) = 5x/sqrt(Re_x)</text>')

    # velocity profiles at a few stations
    for x in (0.15, 0.4, 0.7):
        delta = bl_thickness(U, x, NU)
        xc = X(x)
        prof = []
        m = 24
        for k in range(m + 1):
            eta = k / m
            u = _blasius_profile(eta)
            prof.append(f"{xc + u * 34:.1f},{plate_y - eta * delta * vscale:.1f}")
        parts.append(f'<polyline points="{" ".join(prof)}" fill="none" stroke="#06d6a0" stroke-width="1.8"/>')
        # baseline of the profile
        parts.append(f'<line x1="{xc:.1f}" y1="{plate_y:.1f}" x2="{xc:.1f}" '
                     f'y2="{Yd(delta):.1f}" stroke="#06d6a0" stroke-width="0.7" opacity="0.4"/>')

    # transition marker
    xt = transition_distance(U, NU)
    if xt < L:
        parts.append(f'<line x1="{X(xt):.1f}" y1="{pad+70:.1f}" x2="{X(xt):.1f}" y2="{plate_y:.1f}" '
                     f'stroke="#ff6b6b" stroke-width="1.5" stroke-dasharray="5 4"/>')
        parts.append(f'<text x="{X(xt):.1f}" y="{pad+66:.1f}" fill="#ff6b6b" font-size="11" '
                     f'text-anchor="middle">transition Re_x ~ 5e5 (x = {xt:.2f} m)</text>')

    # skin-friction curve below
    py0, py1 = size * 0.72, size - pad
    cf0 = skin_friction_local(U, 0.02, NU)
    def CY(cf):
        return py1 - (cf / cf0) * (py1 - py0)
    cpts = []
    for i in range(1, n + 1):
        x = L * i / n
        cpts.append(f"{X(x):.1f},{CY(skin_friction_local(U, x, NU)):.1f}")
    parts.append(f'<polyline points="{" ".join(cpts)}" fill="none" stroke="#ff922b" stroke-width="2.2"/>')
    parts.append(f'<text x="{x0+4:.1f}" y="{py0+12:.1f}" fill="#ff922b" font-size="11">'
                 f'skin friction c_f ~ 1/sqrt(x) (peaks at the leading edge)</text>')
    parts.append(f'<text x="{x1-4:.1f}" y="{plate_y+18:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="end">distance x along the plate</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
