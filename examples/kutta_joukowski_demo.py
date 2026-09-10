"""Demo: Kutta-Joukowski -- lift is circulation, and spin makes a ball curve.

Prints the thin-airfoil lift coefficient versus angle of attack, a light-aircraft lift
budget, and the Magnus side-force on a spinning ball, then draws the 2 pi lift-slope line
(with the real-airfoil stall departure) alongside the induced-drag penalty versus aspect
ratio.

    python examples/kutta_joukowski_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kutta_joukowski import (lift_coefficient, lift_force, magnus_force,  # noqa: E402
                             induced_drag_coefficient, thin_airfoil_circulation,
                             kutta_joukowski_lift)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Kutta-Joukowski: L' = rho U Gamma, and thin-airfoil c_l = 2 pi alpha\n")
    print(f"  {'angle of attack':>16}{'c_l (thin)':>12}")
    for deg in (0, 2, 5, 8, 12):
        print(f"  {deg:>13} deg{lift_coefficient(math.radians(deg)):>12.3f}")

    print("\n  Light aircraft (20 m^2 wing, c_l = 0.5, 50 m/s):")
    L = lift_force(0.5, 50.0, 20.0)
    print(f"    lift = {L/1000:.1f} kN  (holds ~{L/9.81/1000:.1f} tonnes)")

    print("\n  Magnus effect on a spinning ball (side force):")
    print(f"  {'ball':<16}{'spin':>10}{'speed':>9}{'side force':>13}")
    for name, r, rpm, U, L_len in (("tennis topspin", 0.033, 3000, 25.0, 0.066),
                                   ("football curl", 0.11, 600, 25.0, 0.22),
                                   ("golf backspin", 0.021, 3000, 60.0, 0.043)):
        omega = 2 * math.pi * rpm / 60.0
        F = magnus_force(r, omega, U, L_len)
        print(f"  {name:<16}{rpm:>7} rpm{U:>7.0f} m/s{F:>11.2f} N")

    print("\n  Lift comes from the circulation the wing sets via the Kutta condition, not any")
    print("  equal-transit myth. Finite wings trail vortices and pay induced drag c_l^2/(pi AR e),")
    print("  which is why soaring birds and gliders wear long, high-aspect-ratio wings.")

    _svg(os.path.join(outdir, "kutta_joukowski.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'kutta_joukowski.svg')}")


def _svg(path, size=720, pad=76):
    # Top: c_l vs alpha (2pi slope + stall). Bottom: induced drag vs aspect ratio.
    x0, x1 = pad, size - pad
    mid = size * 0.52

    # --- lift curve ---
    a_max = 18.0     # degrees
    ty0, ty1 = mid - 30, pad + 40
    def LX(deg):
        return x0 + deg / a_max * (x1 - x0)
    cl_max = 2.0
    def LY(cl):
        return ty0 - cl / cl_max * (ty0 - ty1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="34" fill="#e6edf3" font-size="18">Lift is circulation: c_l = 2 pi alpha</text>',
        f'<text x="20" y="52" fill="#8b949e" font-size="12">'
        f'the thin-airfoil lift-slope, and the stall where a real wing departs from it</text>',
    ]
    parts.append(f'<line x1="{x0}" y1="{ty0}" x2="{x1}" y2="{ty0}" stroke="#8b949e" stroke-width="1.4"/>')
    parts.append(f'<line x1="{x0}" y1="{ty0}" x2="{x0}" y2="{ty1}" stroke="#8b949e" stroke-width="1.4"/>')
    for deg in range(0, 19, 3):
        gx = LX(deg)
        parts.append(f'<line x1="{gx:.1f}" y1="{ty0:.1f}" x2="{gx:.1f}" y2="{ty0+4:.1f}" stroke="#8b949e"/>')
        parts.append(f'<text x="{gx:.1f}" y="{ty0+16:.1f}" fill="#8b949e" font-size="10" '
                     f'text-anchor="middle">{deg}</text>')
    for cl in (0.5, 1.0, 1.5, 2.0):
        gy = LY(cl)
        parts.append(f'<text x="{x0-8:.1f}" y="{gy+4:.1f}" fill="#8b949e" font-size="10" '
                     f'text-anchor="end">{cl:.1f}</text>')
    # 2 pi ideal line
    ideal = [f"{LX(d):.1f},{LY(lift_coefficient(math.radians(d))):.1f}" for d in range(0, 19)]
    parts.append(f'<polyline points="{" ".join(ideal)}" fill="none" stroke="#4dabf7" '
                 f'stroke-width="2.4" stroke-dasharray="6 4"/>')
    parts.append(f'<text x="{LX(13):.1f}" y="{LY(lift_coefficient(math.radians(13)))-6:.1f}" '
                 f'fill="#4dabf7" font-size="11">ideal 2 pi alpha</text>')
    # realistic curve with stall near 15 deg
    real = []
    for d in range(0, 19):
        cl = lift_coefficient(math.radians(d))
        if d > 12:                       # roll off toward stall
            cl = lift_coefficient(math.radians(12)) * (1.0 - 0.12 * (d - 12)) + 0.1 * (d <= 15)
            cl = max(0.7, min(cl, 1.5))
        real.append(f"{LX(d):.1f},{LY(cl):.1f}")
    parts.append(f'<polyline points="{" ".join(real)}" fill="none" stroke="#06d6a0" stroke-width="2.6"/>')
    parts.append(f'<text x="{LX(15):.1f}" y="{LY(1.5)-6:.1f}" fill="#06d6a0" font-size="11" '
                 f'text-anchor="middle">stall</text>')
    parts.append(f'<text x="{(x0+x1)/2:.1f}" y="{ty0+30:.1f}" fill="#8b949e" font-size="11" '
                 f'text-anchor="middle">angle of attack (deg)</text>')

    # --- induced drag vs aspect ratio ---
    by0, by1 = size - pad, mid + 40
    def BX(ar):
        return x0 + (ar - 2.0) / (25.0 - 2.0) * (x1 - x0)
    cdi_max = induced_drag_coefficient(0.6, 2.0)
    def BY(cdi):
        return by0 - cdi / cdi_max * (by0 - by1)
    parts.append(f'<line x1="{x0}" y1="{by0}" x2="{x1}" y2="{by0}" stroke="#8b949e" stroke-width="1.4"/>')
    parts.append(f'<line x1="{x0}" y1="{by0}" x2="{x0}" y2="{by1}" stroke="#8b949e" stroke-width="1.4"/>')
    dpts = []
    ar = 2.0
    while ar <= 25.0:
        dpts.append(f"{BX(ar):.1f},{BY(induced_drag_coefficient(0.6, ar)):.1f}")
        ar += 0.5
    parts.append(f'<polyline points="{" ".join(dpts)}" fill="none" stroke="#ff922b" stroke-width="2.6"/>')
    for ar, lbl in ((5.0, "fighter"), (10.0, "airliner"), (22.0, "glider")):
        parts.append(f'<circle cx="{BX(ar):.1f}" cy="{BY(induced_drag_coefficient(0.6, ar)):.1f}" '
                     f'r="4" fill="#ffd43b"/>')
        parts.append(f'<text x="{BX(ar)+7:.1f}" y="{BY(induced_drag_coefficient(0.6, ar))-6:.1f}" '
                     f'fill="#ffd43b" font-size="10">{lbl}</text>')
    for ar in (5, 10, 15, 20, 25):
        parts.append(f'<text x="{BX(ar):.1f}" y="{by0+16:.1f}" fill="#8b949e" font-size="10" '
                     f'text-anchor="middle">{ar}</text>')
    parts.append(f'<text x="{(x0+x1)/2:.1f}" y="{by0+30:.1f}" fill="#8b949e" font-size="11" '
                 f'text-anchor="middle">aspect ratio (induced drag c_di at c_l=0.6)</text>')
    parts.append(f'<text x="{x0+90:.1f}" y="{by1+4:.1f}" fill="#ff922b" font-size="11">'
                 f'long thin wings pay less induced drag</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
