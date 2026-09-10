"""Demo: the Jeans instability -- the threshold for gravitational collapse.

Plots the dispersion relation omega^2(k): positive (stable sound waves) at short
wavelength, negative (collapse) at long wavelength, crossing zero at the Jeans
wavenumber. Also shows the growth rate of the unstable modes. Renders both to SVG.

    python examples/jeans_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from jeans import (jeans_wavenumber, jeans_length, jeans_mass,  # noqa: E402
                   omega_squared, growth_rate, free_fall_time)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    cs, rho0, G = 1.0, 1.0, 1.0
    kJ = jeans_wavenumber(cs, rho0, G)

    print("Jeans instability: the collapse threshold (c_s = rho0 = G = 1)\n")
    print(f"  Jeans wavenumber k_J : {kJ:.4f}")
    print(f"  Jeans length         : {jeans_length(cs, rho0, G):.4f}")
    print(f"  Jeans mass           : {jeans_mass(cs, rho0, G):.4f}")
    print(f"  free-fall time       : {free_fall_time(rho0, G):.4f}\n")

    print(f"  {'k/k_J':>7}{'omega^2':>12}{'behaviour':>16}")
    print("  " + "-" * 35)
    for frac in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
        k = frac * kJ
        w2 = omega_squared(k, cs, rho0, G)
        beh = "collapse" if w2 < 0 else ("marginal" if abs(w2) < 1e-9 else "sound wave")
        print(f"  {frac:>7.2f}{w2:>12.3f}{beh:>16}")
    print("\n  Below k_J (long wavelength / large cloud) gravity beats pressure and")
    print("  the gas collapses -- the birth of a star. Above it, pressure wins and")
    print("  the perturbation is just a sound wave.")

    ks = [0.02 * kJ * i for i in range(1, 130)]
    w2s = [omega_squared(k, cs, rho0, G) for k in ks]
    grs = [growth_rate(k, cs, rho0, G) for k in ks]
    _svg(ks, w2s, grs, kJ, os.path.join(outdir, "jeans.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'jeans.svg')}")


def _svg(ks, w2s, grs, kJ, path, size=720, pad=60):
    kmax = ks[-1]
    lo = min(min(w2s), min(grs))
    hi = max(max(w2s), max(grs))

    def sx(k):
        return pad + k / kmax * (size - 2 * pad)

    def sy(v):
        return size - pad - (v - lo) / (hi - lo) * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        # zero line and k_J line
        f'<line x1="{pad}" y1="{sy(0):.1f}" x2="{size-pad}" y2="{sy(0):.1f}" stroke="#30363d"/>',
        f'<line x1="{sx(kJ):.1f}" y1="{pad}" x2="{sx(kJ):.1f}" y2="{size-pad}" '
        f'stroke="#e9c46a" stroke-dasharray="4,4"/>',
        f'<text x="{sx(kJ)+6:.1f}" y="{pad+14}" fill="#e9c46a" font-size="12">k_J</text>',
    ]
    w2poly = " ".join(f"{sx(ks[i]):.1f},{sy(w2s[i]):.1f}" for i in range(len(ks)))
    grpoly = " ".join(f"{sx(ks[i]):.1f},{sy(grs[i]):.1f}" for i in range(len(ks)))
    parts.append(f'<polyline points="{w2poly}" fill="none" stroke="#4cc9f0" stroke-width="1.8"/>')
    parts.append(f'<polyline points="{grpoly}" fill="none" stroke="#ff006e" stroke-width="1.8"/>')
    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Jeans dispersion: omega^2 (blue) and growth rate (pink)</text>')
    parts.append(f'<text x="{pad}" y="{size-pad+24}" fill="#8b949e" font-size="11">'
                 f'left of k_J: omega^2&lt;0, collapse. right: sound waves. wavenumber k -&gt;</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
