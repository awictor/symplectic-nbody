"""Demo: the main sequence and the Hertzsprung-Russell diagram.

Shows the mass-luminosity relation and main-sequence lifetimes across stellar
masses, then renders the main sequence on an HR diagram (luminosity vs effective
temperature, temperature increasing to the left, as astronomers plot it) to SVG.

    python examples/main_sequence_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from main_sequence import (luminosity, lifetime_gyr,  # noqa: E402
                           effective_temperature)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("The main sequence: mass sets luminosity, temperature, and lifetime\n")
    print(f"  {'mass (M_sun)':>12}{'L (L_sun)':>12}{'T_eff (K)':>11}{'lifetime':>14}")
    print("  " + "-" * 49)
    for M in (0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0):
        t = lifetime_gyr(M)
        tstr = f"{t*1000:.0f} Myr" if t < 1 else f"{t:.1f} Gyr"
        print(f"  {M:>12.1f}{luminosity(M):>12.1f}{effective_temperature(M):>11.0f}{tstr:>14}")
    print("\n  L ~ M^3.5, so massive O/B stars are millions of times brighter but")
    print("  burn out in a few Myr, while red dwarfs sip fuel for hundreds of Gyr.")
    print("  Plotting L vs T gives the main sequence -- the backbone of the HR diagram.")

    _svg(os.path.join(outdir, "main_sequence.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'main_sequence.svg')}")


def _svg(path, size=720, pad=64):
    masses = [10 ** (-0.6 + 0.02 * i) for i in range(0, 91)]  # ~0.25 .. ~50 Msun
    Ts = [effective_temperature(M) for M in masses]
    Ls = [luminosity(M) for M in masses]
    lT = [math.log10(T) for T in Ts]
    lL = [math.log10(L) for L in Ls]
    Tmin, Tmax = min(lT), max(lT)
    Lmin, Lmax = min(lL), max(lL)

    # HR convention: temperature increases to the LEFT
    def sx(x):
        return pad + (Tmax - x) / (Tmax - Tmin) * (size - 2 * pad)

    def sy(y):
        return size - pad - (y - Lmin) / (Lmax - Lmin) * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    poly = " ".join(f"{sx(lT[i]):.1f},{sy(lL[i]):.1f}" for i in range(len(masses)))
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#ffbe0b" stroke-width="2.4"/>')
    # mark the Sun
    ls = math.log10(effective_temperature(1.0))
    parts.append(f'<circle cx="{sx(ls):.1f}" cy="{sy(0):.1f}" r="5" fill="#f4a261"/>')
    parts.append(f'<text x="{sx(ls)+8:.1f}" y="{sy(0):.1f}" fill="#f4a261" font-size="12">Sun</text>')
    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'The main sequence (HR diagram)</text>')
    parts.append(f'<text x="{pad}" y="{size-pad+24}" fill="#8b949e" font-size="11">'
                 f'log10 T_eff -- hot/blue to the LEFT (astronomer convention)</text>')
    parts.append(f'<text x="{pad-10}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'log10 L / L_sun</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
