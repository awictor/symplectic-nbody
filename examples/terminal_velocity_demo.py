"""Demo: terminal velocity across the Stokes and quadratic-drag regimes.

Prints terminal speeds for objects from fog droplets to skydivers, then draws
terminal velocity vs particle radius for falling spheres, showing the Stokes v ~ r^2
line at small sizes bending to the quadratic v ~ sqrt(r) at large sizes.

    python examples/terminal_velocity_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from terminal_velocity import (terminal_velocity, reynolds_number,  # noqa: E402
                               stokes_velocity, sphere_terminal_velocity,
                               RHO_AIR)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Terminal velocity: drag balances gravity (regime set by Reynolds number)\n")
    print(f"  {'object':>18}{'v_term':>14}{'Re':>12}{'regime':>12}")
    print("  " + "-" * 56)
    # (name, radius, density) falling spheres in air
    drops = [
        ("fog droplet 10um", 1e-5, 1000.0),
        ("drizzle 100um", 1e-4, 1000.0),
        ("raindrop 2mm", 2e-3, 1000.0),
        ("hailstone 1cm", 1e-2, 900.0),
        ("steel ball 1cm", 1e-2, 7800.0),
    ]
    for name, r, rho_p in drops:
        vs = stokes_velocity(r, rho_p)
        Re_s = reynolds_number(vs, 2 * r)
        if Re_s < 1.0:
            v, Re, regime = vs, Re_s, "Stokes"
        else:
            v = sphere_terminal_velocity(r, rho_p)
            Re, regime = reynolds_number(v, 2 * r), "quadratic"
        vstr = f"{v*1000:.2f} mm/s" if v < 0.5 else f"{v:.1f} m/s"
        print(f"  {name:>18}{vstr:>14}{Re:>12.2g}{regime:>12}")

    vsky = terminal_velocity(75.0, 0.5, 1.0)
    print(f"\n  {'skydiver (belly)':>18}{vsky:>10.0f} m/s  (~{vsky*3.6:.0f} km/h)")

    print("\n  Stokes drag (viscous) gives v ~ r^2, so a fog droplet 200x smaller than")
    print("  a raindrop falls ~40000x slower -- effectively floating. Big drops cross")
    print("  into quadratic drag where v ~ sqrt(r), so a hailstone and a steel ball of")
    print("  the same size differ only by sqrt(density). The regime boundary is Re ~ 1.")

    _svg(os.path.join(outdir, "terminal_velocity.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'terminal_velocity.svg')}")


def _v_of_r(r, rho_p=1000.0):
    """Pick the regime by Reynolds number and return the terminal velocity."""
    vs = stokes_velocity(r, rho_p)
    if reynolds_number(vs, 2 * r) < 1.0:
        return vs
    return sphere_terminal_velocity(r, rho_p)


def _svg(path, size=720, pad=70):
    rs = [10 ** (-7 + 0.08 * i) for i in range(0, 76)]   # 0.1 um .. ~few cm
    vs = [_v_of_r(r) for r in rs]
    lx = [math.log10(r) for r in rs]
    ly = [math.log10(v) for v in vs]
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
    # mark the Re=1 regime boundary (find where slope changes)
    for i in range(1, len(rs)):
        if reynolds_number(stokes_velocity(rs[i], 1000.0), 2 * rs[i]) >= 1.0:
            bx = sx(lx[i])
            parts.append(f'<line x1="{bx:.1f}" y1="{pad}" x2="{bx:.1f}" y2="{size-pad}" '
                         f'stroke="#ff6b6b" stroke-width="1.2" stroke-dasharray="4 4"/>')
            parts.append(f'<text x="{bx+5:.1f}" y="{pad+16:.1f}" fill="#ff6b6b" '
                         f'font-size="11">Re ~ 1 (Stokes | quadratic)</text>')
            break

    poly = " ".join(f"{sx(lx[i]):.1f},{sy(ly[i]):.1f}" for i in range(len(rs)))
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#4dabf7" stroke-width="2.6"/>')

    labels = [("fog", 1e-5), ("raindrop", 2e-3), ("hail", 1e-2)]
    for name, r in labels:
        v = _v_of_r(r)
        px = sx(math.log10(r))
        py = sy(math.log10(v))
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.5" fill="#ffd43b"/>')
        parts.append(f'<text x="{px+7:.1f}" y="{py+4:.1f}" fill="#ffd43b" '
                     f'font-size="10">{name}</text>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Terminal velocity vs particle radius (water in air)</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'Stokes v ~ r^2 for small drops, quadratic v ~ sqrt(r) for big ones</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+24}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">log10 radius (m) -&gt;</text>')
    parts.append(f'<text x="{pad-16}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'log10 terminal velocity (m/s)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
