"""Demo: Bondi accretion -- how compact objects feed on ambient gas.

Shows the Bondi accretion rate for a black hole across gas temperatures and
masses, and renders the steep temperature dependence (rate ~ c_s^{-3} ~ T^{-3/2})
to a log-log SVG.

    python examples/bondi_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bondi import (bondi_radius, bondi_rate, accretion_luminosity,  # noqa: E402
                   sound_speed, M_SUN, YEAR)

RHO = 1.67e-21  # ~1 proton/cc


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Bondi accretion (10 M_sun object, n ~ 1 /cm^3)\n")
    print(f"  {'gas T (K)':>10}{'c_s (km/s)':>12}{'r_B (AU)':>10}{'Mdot (Msun/yr)':>16}")
    print("  " + "-" * 48)
    for T in (100, 1000, 1e4, 1e6, 1e7):
        cs = sound_speed(T)
        M = 10 * M_SUN
        mdot = bondi_rate(M, RHO, cs) / M_SUN * YEAR
        print(f"  {T:>10.0e}{cs/1e3:>12.1f}{bondi_radius(M, cs)/1.496e11:>10.1f}{mdot:>16.2e}")
    print("\n  Rate ~ M^2 rho / c_s^3, so accretion runs away with mass and is far")
    print("  stronger in cold gas: a black hole in a 100 K molecular cloud accretes")
    print("  millions of times faster than one in hot 10^7 K coronal gas.")

    _svg(os.path.join(outdir, "bondi.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'bondi.svg')}")


def _svg(path, size=720, pad=64):
    Ts = [10 ** (2 + 0.05 * i) for i in range(0, 101)]  # 100 K .. 1e7 K
    M = 10 * M_SUN
    rates = [bondi_rate(M, RHO, sound_speed(T)) / M_SUN * YEAR for T in Ts]
    lT = [math.log10(T) for T in Ts]
    lr = [math.log10(r) for r in rates]
    Tmin, Tmax = lT[0], lT[-1]
    rmin, rmax = min(lr), max(lr)

    def sx(x):
        return pad + (x - Tmin) / (Tmax - Tmin) * (size - 2 * pad)

    def sy(y):
        return size - pad - (y - rmin) / (rmax - rmin) * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    poly = " ".join(f"{sx(lT[i]):.1f},{sy(lr[i]):.1f}" for i in range(len(Ts)))
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#4cc9f0" stroke-width="2.2"/>')
    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Bondi accretion rate vs gas temperature</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'Mdot ~ c_s^-3 ~ T^-3/2: cold gas is devoured, hot gas barely touched</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+22}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">log10 gas temperature (K) -&gt;</text>')
    parts.append(f'<text x="{pad-10}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'log10 Mdot (M_sun/yr)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
