"""Demo: the Kelvin-Helmholtz timescale and why the Sun needs nuclear fusion.

Contrasts the gravitational (Kelvin-Helmholtz) timescale with the nuclear
main-sequence lifetime across stellar masses, showing that gravity alone gives
only ~30 Myr for the Sun -- far too short. Renders both timescales vs mass to a
log-log SVG.

    python examples/kelvin_helmholtz_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kelvin_helmholtz import kh_time_solar_units, solar_kh_time_myr  # noqa: E402
from main_sequence import lifetime_gyr  # noqa: E402


def radius_rsun(M):
    return M ** 0.8


def lum_lsun(M):
    return M ** 3.5


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Kelvin-Helmholtz (gravity) vs nuclear timescales\n")
    print(f"  Sun's KH time: {solar_kh_time_myr():.0f} Myr -- vs Earth's 4500 Myr age.")
    print(f"  Gravity alone runs the Sun for only ~30 Myr, so it MUST fuse hydrogen.\n")
    print(f"  {'mass (M_sun)':>12}{'t_KH (Myr)':>14}{'t_nuclear (Myr)':>18}")
    print("  " + "-" * 44)
    for M in (0.5, 1.0, 2.0, 5.0, 10.0):
        tkh = kh_time_solar_units(M, radius_rsun(M), lum_lsun(M))
        tnuc = lifetime_gyr(M) * 1000.0
        print(f"  {M:>12.1f}{tkh:>14.2f}{tnuc:>18.1f}")
    print("\n  The nuclear lifetime dwarfs the KH time at every mass -- fusion, not")
    print("  contraction, powers stars. The KH time is instead how long a")
    print("  protostar contracts before fusion ignites.")

    _svg(os.path.join(outdir, "kelvin_helmholtz.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'kelvin_helmholtz.svg')}")


def _svg(path, size=720, pad=64):
    masses = [10 ** (-0.3 + 0.015 * i) for i in range(0, 101)]  # ~0.5 .. ~16 Msun
    kh = [kh_time_solar_units(M, radius_rsun(M), lum_lsun(M)) for M in masses]
    nuc = [lifetime_gyr(M) * 1000.0 for M in masses]
    lm = [math.log10(M) for M in masses]
    lkh = [math.log10(t) for t in kh]
    lnuc = [math.log10(t) for t in nuc]
    mmin, mmax = lm[0], lm[-1]
    ymin = min(min(lkh), min(lnuc))
    ymax = max(max(lkh), max(lnuc))

    def sx(x):
        return pad + (x - mmin) / (mmax - mmin) * (size - 2 * pad)

    def sy(y):
        return size - pad - (y - ymin) / (ymax - ymin) * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    khp = " ".join(f"{sx(lm[i]):.1f},{sy(lkh[i]):.1f}" for i in range(len(masses)))
    nucp = " ".join(f"{sx(lm[i]):.1f},{sy(lnuc[i]):.1f}" for i in range(len(masses)))
    parts.append(f'<polyline points="{khp}" fill="none" stroke="#e63946" stroke-width="2"/>')
    parts.append(f'<polyline points="{nucp}" fill="none" stroke="#4cc9f0" stroke-width="2"/>')
    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Timescales vs stellar mass (log-log, Myr)</text>')
    parts.append(f'<text x="{pad+14}" y="{pad+8}" fill="#4cc9f0" font-size="12">'
                 f'nuclear (main-sequence) lifetime</text>')
    parts.append(f'<text x="{pad+14}" y="{pad+26}" fill="#e63946" font-size="12">'
                 f'Kelvin-Helmholtz (gravity) time -- far shorter</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+22}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">log10 mass (M_sun) -&gt;</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
