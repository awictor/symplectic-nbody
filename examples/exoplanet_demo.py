"""Demo: exoplanet detection by transits and radial velocity.

Computes transit depths and RV semi-amplitudes for several planets, and renders a
schematic transit light curve (a box dip of depth (R_p/R_star)^2) to SVG.

    python examples/exoplanet_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from exoplanet import (transit_depth, orbital_period, rv_semi_amplitude,  # noqa: E402
                       M_SUN, R_SUN, M_JUP, R_JUP, M_EARTH, R_EARTH, AU, YEAR)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Exoplanet detection: transit depth and radial-velocity wobble\n")
    print(f"  {'planet':<16}{'a (AU)':>8}{'depth':>12}{'RV K (m/s)':>13}{'period':>12}")
    print("  " + "-" * 61)
    cases = [
        ("hot Jupiter", M_JUP, R_JUP, 0.05),
        ("Jupiter", M_JUP, R_JUP, 5.204),
        ("Earth", M_EARTH, R_EARTH, 1.0),
    ]
    for name, m, R, a_au in cases:
        a = a_au * AU
        P = orbital_period(a, M_SUN)
        Pstr = f"{P/86400:.1f} d" if P < YEAR else f"{P/YEAR:.1f} yr"
        print(f"  {name:<16}{a_au:>8.3f}{transit_depth(R, R_SUN):>12.2e}"
              f"{rv_semi_amplitude(m, M_SUN, a):>13.2f}{Pstr:>12}")
    print("\n  A transiting Jupiter dims its star ~1%; an Earth only 0.008%. Jupiter")
    print("  wobbles the Sun 12 m/s, Earth just 9 cm/s. Hot Jupiters -- big, close,")
    print("  fast -- give the strongest signals, which is why they were found first.")

    _svg(transit_depth(R_JUP, R_SUN), os.path.join(outdir, "exoplanet.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'exoplanet.svg')}")


def _svg(depth, path, size=720, pad=64):
    # schematic transit light curve: flat, dip to (1-depth), flat
    n = 400
    flux = []
    for i in range(n):
        x = i / (n - 1)
        # box dip with soft ingress/egress between 0.4 and 0.6
        if 0.42 < x < 0.58:
            f = 1.0 - depth
        elif 0.40 <= x <= 0.42:
            f = 1.0 - depth * (x - 0.40) / 0.02
        elif 0.58 <= x <= 0.60:
            f = 1.0 - depth * (0.60 - x) / 0.02
        else:
            f = 1.0
        flux.append(f)
    # exaggerate the depth visually (real Jupiter dip is 1%)
    fmin = 1.0 - depth

    def sx(i):
        return pad + i / (n - 1) * (size - 2 * pad)

    def sy(f):
        lo, hi = fmin - depth * 0.3, 1.0 + depth * 0.3
        return size - pad - (f - lo) / (hi - lo) * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    poly = " ".join(f"{sx(i):.1f},{sy(flux[i]):.1f}" for i in range(n))
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#ffbe0b" stroke-width="2.2"/>')
    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Transit light curve (Jupiter across the Sun)</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'brightness dips by (R_p/R_star)^2 = {depth*100:.2f}% during transit</text>')
    parts.append(f'<text x="{pad}" y="{size-pad+24}" fill="#8b949e" font-size="11">'
                 f'time -&gt;</text>')
    parts.append(f'<text x="{pad-10}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'relative brightness</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
