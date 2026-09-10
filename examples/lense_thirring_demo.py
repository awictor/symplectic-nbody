"""Demo: frame-dragging and geodetic precession (Gravity Probe B).

Reproduces the two general-relativistic gyroscope precessions Gravity Probe B
measured, shows their steep radius dependence, and renders both rates vs orbital
altitude to a log-log SVG.

    python examples/lense_thirring_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lense_thirring import (geodetic_mas_per_year, frame_dragging_mas_per_year,  # noqa: E402
                            gravity_probe_b_altitude, M_EARTH, J_EARTH, R_EARTH)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    r = gravity_probe_b_altitude()
    print("Gravity Probe B: two relativistic gyroscope precessions\n")
    print(f"  {'effect':<26}{'predicted':>14}{'measured':>12}")
    print("  " + "-" * 52)
    print(f"  {'geodetic (de Sitter)':<26}{geodetic_mas_per_year(M_EARTH, r):>11.0f} mas/yr"
          f"{'6602':>12}")
    print(f"  {'frame-dragging (LT)':<26}{frame_dragging_mas_per_year(J_EARTH, r):>11.1f} mas/yr"
          f"{'37.2':>12}")
    print("\n  Geodetic precession comes from the curvature of space the gyro is")
    print("  carried through; frame-dragging comes from Earth's rotation twisting")
    print("  spacetime. The latter is ~180x smaller -- it took a dedicated")
    print("  experiment with near-perfect gyroscopes to measure it.")

    _svg(os.path.join(outdir, "lense_thirring.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'lense_thirring.svg')}")


def _svg(path, size=720, pad=64):
    # rates vs altitude (log-log)
    alts = [R_EARTH * (1.0 + 0.05 * i) for i in range(1, 120)]  # up to ~7 R_E
    geo = [geodetic_mas_per_year(M_EARTH, r) for r in alts]
    lt = [frame_dragging_mas_per_year(J_EARTH, r) for r in alts]
    lr = [math.log10(r / R_EARTH) for r in alts]
    lgeo = [math.log10(g) for g in geo]
    llt = [math.log10(x) for x in lt]

    rmin, rmax = lr[0], lr[-1]
    ymin = min(min(lgeo), min(llt))
    ymax = max(max(lgeo), max(llt))

    def sx(x):
        return pad + (x - rmin) / (rmax - rmin) * (size - 2 * pad)

    def sy(y):
        return size - pad - (y - ymin) / (ymax - ymin) * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    gpoly = " ".join(f"{sx(lr[i]):.1f},{sy(lgeo[i]):.1f}" for i in range(len(alts)))
    lpoly = " ".join(f"{sx(lr[i]):.1f},{sy(llt[i]):.1f}" for i in range(len(alts)))
    parts.append(f'<polyline points="{gpoly}" fill="none" stroke="#4cc9f0" stroke-width="2"/>')
    parts.append(f'<polyline points="{lpoly}" fill="none" stroke="#ff006e" stroke-width="2"/>')
    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Gyroscope precession vs orbit radius (log-log)</text>')
    parts.append(f'<text x="{pad+14}" y="{pad+8}" fill="#4cc9f0" font-size="12">'
                 f'geodetic ~ r^-5/2</text>')
    parts.append(f'<text x="{pad+14}" y="{pad+26}" fill="#ff006e" font-size="12">'
                 f'frame-dragging ~ r^-3 (smaller, steeper)</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+22}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">log10 (r / R_Earth) -&gt;</text>')
    parts.append(f'<text x="{pad-10}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'log10 precession (mas/yr)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
