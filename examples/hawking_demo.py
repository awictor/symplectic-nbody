"""Demo: Hawking radiation and black-hole thermodynamics.

Tabulates temperature, entropy, and evaporation time across black-hole masses
from a primordial hole to a supermassive one, finds the primordial mass
evaporating in a Hubble time, and renders T(M) and t_evap(M) to a log-log SVG.

    python examples/hawking_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hawking import (hawking_temperature, evaporation_time, entropy_over_kb,  # noqa: E402
                     mass_evaporating_in, M_SUN, YEAR)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Hawking radiation: black holes are not black\n")
    t_universe = 13.8e9 * YEAR
    M_now = mass_evaporating_in(t_universe)
    print(f"  primordial mass evaporating in a Hubble time: {M_now:.2e} kg")
    print(f"    (about the mass of a large asteroid, in a horizon ~1e-16 m across)\n")

    cases = [
        ("primordial (now)", M_now),
        ("1 solar mass", M_SUN),
        ("Sgr A* (4e6 Msun)", 4e6 * M_SUN),
        ("M87* (6.5e9 Msun)", 6.5e9 * M_SUN),
    ]
    print(f"  {'object':<22}{'T (K)':>12}{'t_evap (yr)':>14}{'S/k_B':>12}")
    print("  " + "-" * 60)
    for name, M in cases:
        print(f"  {name:<22}{hawking_temperature(M):>12.2e}"
              f"{evaporation_time(M)/YEAR:>14.2e}{entropy_over_kb(M):>12.2e}")
    print("\n  Bigger holes are COLDER and live longer (T ~ 1/M, t_evap ~ M^3).")
    print("  A stellar black hole is nanokelvin-cold and effectively eternal;")
    print("  its entropy exceeds that of everything else in the observable universe.")

    _svg(os.path.join(outdir, "hawking.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'hawking.svg')}")


def _svg(path, size=720, pad=64):
    masses = [10 ** (8 + 0.4 * i) for i in range(0, 65)]  # 1e8 .. ~1e33 kg
    logm = [math.log10(m) for m in masses]
    logT = [math.log10(hawking_temperature(m)) for m in masses]
    logt = [math.log10(evaporation_time(m) / YEAR) for m in masses]

    mmin, mmax = logm[0], logm[-1]
    ymin = min(min(logT), min(logt))
    ymax = max(max(logT), max(logt))

    def sx(lm):
        return pad + (lm - mmin) / (mmax - mmin) * (size - 2 * pad)

    def sy(y):
        return size - pad - (y - ymin) / (ymax - ymin) * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    Tpoly = " ".join(f"{sx(logm[i]):.1f},{sy(logT[i]):.1f}" for i in range(len(masses)))
    tpoly = " ".join(f"{sx(logm[i]):.1f},{sy(logt[i]):.1f}" for i in range(len(masses)))
    parts.append(f'<polyline points="{Tpoly}" fill="none" stroke="#ff006e" stroke-width="2"/>')
    parts.append(f'<polyline points="{tpoly}" fill="none" stroke="#4cc9f0" stroke-width="2"/>')
    # solar-mass marker
    lsun = math.log10(M_SUN)
    parts.append(f'<line x1="{sx(lsun):.1f}" y1="{pad}" x2="{sx(lsun):.1f}" y2="{size-pad}" '
                 f'stroke="#e9c46a" stroke-dasharray="4,4"/>')
    parts.append(f'<text x="{sx(lsun)+6:.1f}" y="{pad+14}" fill="#e9c46a" font-size="11">1 M_sun</text>')
    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Black-hole thermodynamics vs mass (log-log)</text>')
    parts.append(f'<text x="{pad+14}" y="{pad+8}" fill="#ff006e" font-size="12">'
                 f'log T (K): colder for bigger holes</text>')
    parts.append(f'<text x="{pad+14}" y="{pad+26}" fill="#4cc9f0" font-size="12">'
                 f'log t_evap (yr): longer for bigger holes</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+22}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">log10 mass (kg) -&gt;</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
