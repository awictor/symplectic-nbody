"""Demo: Rutherford scattering off the nucleus.

Prints the cross section at several angles and the head-on closest approach for alpha
particles on gold, then draws the 1/sin^4(theta/2) angular distribution -- the curve
whose nonzero back-scattering revealed the atomic nucleus.

    python examples/rutherford_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rutherford import (differential_cross_section, impact_parameter,  # noqa: E402
                        closest_approach, scattering_angle, MEV, FM)

Z_ALPHA, Z_GOLD = 2, 79


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    E = 5 * MEV
    print("Rutherford scattering: dsigma/dOmega ~ 1/sin^4(theta/2)\n")
    print(f"  5 MeV alpha on gold: head-on closest approach = "
          f"{closest_approach(E, Z_ALPHA, Z_GOLD)/FM:.1f} fm\n")
    print(f"  {'angle':>8}{'dsigma/dOmega (rel)':>22}{'impact b (fm)':>16}")
    print("  " + "-" * 46)
    ref = differential_cross_section(math.radians(90), Z_ALPHA, Z_GOLD, E)
    for deg in (10, 30, 60, 90, 120, 150, 179):
        d = differential_cross_section(math.radians(deg), Z_ALPHA, Z_GOLD, E)
        b = impact_parameter(math.radians(deg), Z_ALPHA, Z_GOLD, E) / FM
        print(f"  {deg:>7}d{d/ref:>22.3g}{b:>16.1f}")

    print("\n  The cross section soars at small angles (grazing passes barely deflect)")
    print("  and falls toward back-scattering -- but stays NONZERO at 180 degrees. Those")
    print("  rare hard bounces are impossible off diffuse charge; they proved the atom's")
    print("  positive charge sits in a tiny dense nucleus. The head-on approach distance")
    print("  put an upper bound of tens of femtometres on the nuclear size.")

    _svg(os.path.join(outdir, "rutherford.svg"), E)
    print(f"\n  wrote {os.path.join(outdir, 'rutherford.svg')}")


def _svg(path, E, size=720, pad=72):
    degs = [1.0 + 1.0 * i for i in range(0, 179)]
    ref = differential_cross_section(math.radians(90), Z_ALPHA, Z_GOLD, E)
    ds = [math.log10(differential_cross_section(math.radians(d), Z_ALPHA, Z_GOLD, E) / ref)
          for d in degs]
    xmin, xmax = 0.0, 180.0
    ymin, ymax = min(ds), max(ds)

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
    poly = " ".join(f"{sx(degs[i]):.1f},{sy(ds[i]):.1f}" for i in range(len(degs)))
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#ffd43b" stroke-width="2.6"/>')

    # mark the back-scatter tail (nonzero at 180)
    x180 = sx(179)
    parts.append(f'<circle cx="{x180:.1f}" cy="{sy(ds[-1]):.1f}" r="4" fill="#ff6b6b"/>')
    parts.append(f'<text x="{x180-6:.1f}" y="{sy(ds[-1])-8:.1f}" fill="#ff6b6b" '
                 f'font-size="10" text-anchor="end">nonzero back-scatter -> nucleus!</text>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Rutherford cross section vs scattering angle</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'5 MeV alpha on gold; ~1/sin^4(theta/2), diverging at small angle</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+24}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">scattering angle (deg) -&gt;</text>')
    parts.append(f'<text x="{pad-18}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'log10 dsigma/dOmega (rel. to 90 deg)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
