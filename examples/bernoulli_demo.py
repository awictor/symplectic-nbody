"""Demo: Bernoulli's principle and the Venturi effect.

Prints Pitot airspeeds and Torricelli efflux speeds, then draws a Venturi tube: as the
pipe narrows the flow speeds up and the pressure drops, the trade at the heart of
carburettors, flow meters, and lift.

    python examples/bernoulli_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bernoulli import (pressure_at, venturi_velocity, pitot_airspeed,  # noqa: E402
                       torricelli_speed, dynamic_pressure, RHO_WATER, RHO_AIR)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Bernoulli: P + 1/2 rho v^2 + rho g h = const -- fast flow, low pressure\n")
    print("  Pitot airspeed from dynamic pressure (air):")
    print(f"  {'true airspeed':>16}{'dynamic P (kPa)':>18}")
    for v in (50, 100, 250):
        q = dynamic_pressure(v, RHO_AIR)
        print(f"  {v:>13} m/s{q/1000:>18.2f}")

    print("\n  Torricelli efflux (water jet from a hole at depth h):")
    for h in (1, 5, 20):
        print(f"    depth {h:>3} m  ->  {torricelli_speed(h):.1f} m/s")

    print("\n  A narrowing pipe (Venturi) speeds the flow -- so by Bernoulli its pressure")
    print("  drops right where it is fastest. That suction draws fuel into a carburettor,")
    print("  lets a Venturi meter read flow from a pressure gap, and, with circulation,")
    print("  helps hold an airfoil up. A Pitot tube runs it backwards, stopping the flow")
    print("  to turn its speed into a pressure an aircraft reads as airspeed.")

    _svg(os.path.join(outdir, "bernoulli.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'bernoulli.svg')}")


def _svg(path, size=720, pad=70):
    # Venturi pipe profile: wide -> narrow throat -> wide, plot velocity and pressure
    xs = [i / 200.0 for i in range(0, 201)]   # 0..1 along the pipe
    # area profile: dips in the middle
    def area(x):
        return 1.0 - 0.6 * math.exp(-((x - 0.5) / 0.12) ** 2)
    A1 = area(0.0)
    v1 = 1.0
    P1 = 1.0
    vs = [v1 * A1 / area(x) for x in xs]                    # continuity
    ps = [P1 + 0.5 * (v1 ** 2 - v ** 2) for v in vs]        # Bernoulli (rho=1)

    top = size * 0.30
    mid = size * 0.42
    # pipe walls (area -> half-height)
    def wall(x):
        return mid - area(x) / A1 * (mid - top)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
    ]
    def px(x):
        return pad + x * (size - 2 * pad)
    # draw pipe (mirror walls about mid)
    up = " ".join(f"{px(xs[i]):.1f},{wall(xs[i]):.1f}" for i in range(len(xs)))
    dn = " ".join(f"{px(xs[i]):.1f},{2*mid - wall(xs[i]):.1f}" for i in range(len(xs)))
    parts.append(f'<polyline points="{up}" fill="none" stroke="#8b949e" stroke-width="2"/>')
    parts.append(f'<polyline points="{dn}" fill="none" stroke="#8b949e" stroke-width="2"/>')
    parts.append(f'<text x="{px(0.5):.1f}" y="{top-8:.1f}" fill="#8b949e" '
                 f'font-size="11" text-anchor="middle">throat (narrow, fast, low P)</text>')

    # velocity and pressure curves below
    vy0, vy1 = size * 0.55, size * 0.72
    vmax = max(vs)
    py0, py1 = size * 0.75, size - pad
    pmin, pmax = min(ps), max(ps)
    def vy(v):
        return vy1 - (v / vmax) * (vy1 - vy0)
    def py(p):
        return py1 - (p - pmin) / (pmax - pmin) * (py1 - py0)
    vpoly = " ".join(f"{px(xs[i]):.1f},{vy(vs[i]):.1f}" for i in range(len(xs)))
    ppoly = " ".join(f"{px(xs[i]):.1f},{py(ps[i]):.1f}" for i in range(len(xs)))
    parts.append(f'<polyline points="{vpoly}" fill="none" stroke="#4dabf7" stroke-width="2.4"/>')
    parts.append(f'<polyline points="{ppoly}" fill="none" stroke="#ff6b6b" stroke-width="2.4"/>')
    parts.append(f'<text x="{px(0.02):.1f}" y="{vy0-4:.1f}" fill="#4dabf7" font-size="12">velocity (peaks at throat)</text>')
    parts.append(f'<text x="{px(0.02):.1f}" y="{py0-4:.1f}" fill="#ff6b6b" font-size="12">pressure (dips at throat)</text>')

    parts.append(f'<text x="20" y="34" fill="#e6edf3" font-size="18">'
                 f'The Venturi effect (Bernoulli)</text>')
    parts.append(f'<text x="20" y="52" fill="#8b949e" font-size="12">'
                 f'where the pipe narrows the flow speeds up and the pressure falls</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
