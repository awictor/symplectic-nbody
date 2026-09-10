"""Demo: the Eddington luminosity and how fast black holes can grow.

Tabulates the Eddington luminosity and accretion rate across masses, shows the
Salpeter e-folding time, and renders the exponential growth of an Eddington-
limited black hole (seed -> quasar) to SVG.

    python examples/eddington_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eddington import (eddington_luminosity, eddington_luminosity_solar_units,  # noqa: E402
                       eddington_accretion_rate, salpeter_time, M_SUN, YEAR)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("The Eddington luminosity: the brightness limit of accretion\n")
    print(f"  Salpeter e-folding time: {salpeter_time()/YEAR/1e6:.1f} Myr "
          f"(eta=0.1, Eddington-limited)\n")
    print(f"  {'object':<24}{'M (Msun)':>10}{'L_Edd (L_sun)':>16}{'Mdot (Msun/yr)':>16}")
    print("  " + "-" * 66)
    cases = [("Sun", 1.0), ("stellar BH", 10.0), ("Sgr A*", 4e6), ("quasar", 1e9)]
    for name, m in cases:
        Ls = eddington_luminosity_solar_units(m)
        mdot = eddington_accretion_rate(m * M_SUN) / M_SUN * YEAR
        print(f"  {name:<24}{m:>10.0e}{Ls:>16.2e}{mdot:>16.2e}")

    print("\n  L_Edd is linear in mass and radius-independent. Above it, radiation")
    print("  pressure blows the accreting gas away, so it caps how fast a black")
    print("  hole can grow -- the exponential Salpeter track below.")

    _svg(os.path.join(outdir, "eddington.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'eddington.svg')}")


def _svg(path, size=720, pad=64):
    # exponential growth M(t) = M0 exp(t / t_S), seed 10 Msun to 1e9 Msun
    tS = salpeter_time() / YEAR / 1e6  # Myr
    M0 = 10.0
    t_final = tS * math.log(1e9 / M0)
    ts = [t_final * i / 200 for i in range(201)]
    Ms = [M0 * math.exp(t / tS) for t in ts]           # solar masses
    logM = [math.log10(m) for m in Ms]

    def sx(t):
        return pad + t / t_final * (size - 2 * pad)

    def sy(lm):
        return size - pad - (lm - 1.0) / (9.0 - 1.0) * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    poly = " ".join(f"{sx(ts[i]):.1f},{sy(logM[i]):.1f}" for i in range(len(ts)))
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#ffbe0b" stroke-width="2"/>')
    # 1e9 Msun quasar line
    parts.append(f'<line x1="{pad}" y1="{sy(9):.1f}" x2="{size-pad}" y2="{sy(9):.1f}" '
                 f'stroke="#ff006e" stroke-dasharray="5,4"/>')
    parts.append(f'<text x="{size-pad-4}" y="{sy(9)-6:.1f}" fill="#ff006e" font-size="12" '
                 f'text-anchor="end">1e9 M_sun quasar</text>')
    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Eddington-limited black-hole growth</text>')
    parts.append(f'<text x="{pad}" y="{size-pad+24}" fill="#8b949e" font-size="11">'
                 f'time (Myr); reaching a quasar mass takes ~{t_final:.0f} Myr</text>')
    parts.append(f'<text x="{pad-10}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'log10 mass (M_sun)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
