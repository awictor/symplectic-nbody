"""Demo: the Shakura-Sunyaev accretion disk.

Prints inner-edge, peak and characteristic temperatures plus the Eddington
luminosity for a stellar-mass and a supermassive black hole, then draws the
T ~ r^(-3/4) temperature profiles so the X-ray-hot stellar disk and the UV-bright
quasar disk sit side by side.

    python examples/accretion_disk_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from accretion_disk import (isco_radius, disk_temperature,  # noqa: E402
                            peak_temperature, eddington_luminosity,
                            luminosity, radiative_efficiency, M_SUN, C,
                            ETA_SCHWARZSCHILD)

K_B = 1.380649e-23
KEV = 1.602176634e-16
L_SUN = 3.828e26


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Shakura-Sunyaev disk: T(r) ~ r^(-3/4), L = eta Mdot c^2\n")
    print(f"  radiative efficiency eta = {radiative_efficiency():.3f} "
          f"(vs ~0.007 for H fusion)\n")

    cases = [("stellar-mass BH", 10 * M_SUN), ("supermassive BH", 1e8 * M_SUN)]
    print(f"  {'object':>18}{'r_in (km)':>13}{'peak T (K)':>13}"
          f"{'peak kT':>12}{'L_Edd (Lsun)':>15}")
    print("  " + "-" * 71)
    for name, M in cases:
        mdot = eddington_luminosity(M) / (ETA_SCHWARZSCHILD * C * C)
        rin = isco_radius(M) / 1e3
        Tp = peak_temperature(M, mdot)
        kT = K_B * Tp / KEV
        Ledd = eddington_luminosity(M) / L_SUN
        ktstr = f"{kT:.2f} keV" if kT >= 0.1 else f"{kT*1000:.0f} eV"
        print(f"  {name:>18}{rin:>13.1f}{Tp:>13.2e}{ktstr:>12}{Ledd:>15.2e}")

    print("\n  The stellar-mass disk peaks around a keV -- soft X-rays, which is how")
    print("  black-hole binaries are found. The billion-times-heavier supermassive")
    print("  disk is a thousand times cooler at its edge (T_* ~ M^(-1/4)), peaking")
    print("  in the ultraviolet: the 'big blue bump' of quasar spectra. Both convert")
    print("  ~6% of infalling rest mass to light -- ~8x more than fusion manages.")

    _svg(os.path.join(outdir, "accretion_disk.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'accretion_disk.svg')}")


def _svg(path, size=720, pad=68):
    curves = []
    for name, M, col in [("stellar-mass (10 Msun)", 10 * M_SUN, "#ff922b"),
                         ("supermassive (1e8 Msun)", 1e8 * M_SUN, "#4dabf7")]:
        mdot = eddington_luminosity(M) / (ETA_SCHWARZSCHILD * C * C)
        rin = isco_radius(M)
        xs = [rin * (1.15 ** i) for i in range(1, 60)]
        pts = []
        for r in xs:
            T = disk_temperature(r, M, mdot, rin)
            if T > 0:
                pts.append((r / rin, T))
        curves.append((name, col, pts))

    # x = r / r_in (log), y = T (log)
    allx = [x for _, _, pts in curves for x, _ in pts]
    ally = [y for _, _, pts in curves for _, y in pts]
    lx = [math.log10(x) for x in allx]
    ly = [math.log10(y) for y in ally]
    xmin, xmax = min(lx), max(lx)
    ymin, ymax = min(ly), max(ly)

    def sx(x):
        return pad + (x - xmin) / (xmax - xmin) * (size - 2 * pad)

    def sy(y):
        return size - pad - (y - ymin) / (ymax - ymin) * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
    ]
    # waveband guide lines: X-ray ~1e7 K, UV ~1e5 K, optical ~1e4 K
    for T_band, label, col in [(1e7, "X-ray", "#f06595"),
                               (1e5, "UV", "#845ef7"),
                               (1e4, "optical", "#22b8cf")]:
        if ymin <= math.log10(T_band) <= ymax:
            yb = sy(math.log10(T_band))
            parts.append(f'<line x1="{pad}" y1="{yb:.1f}" x2="{size-pad}" y2="{yb:.1f}" '
                         f'stroke="{col}" stroke-width="1" stroke-dasharray="2 5" opacity="0.6"/>')
            parts.append(f'<text x="{size-pad-4:.1f}" y="{yb-4:.1f}" fill="{col}" '
                         f'font-size="10" text-anchor="end">{label} ({T_band:.0e} K)</text>')

    parts.append(f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>')
    parts.append(f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>')

    ytop = pad + 22
    for i, (name, col, pts) in enumerate(curves):
        poly = " ".join(f"{sx(math.log10(x)):.1f},{sy(math.log10(y)):.1f}" for x, y in pts)
        parts.append(f'<polyline points="{poly}" fill="none" stroke="{col}" stroke-width="2.4"/>')
        parts.append(f'<text x="{pad+10}" y="{ytop + i*16}" fill="{col}" '
                     f'font-size="12">{name}</text>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Accretion-disk temperature: T ~ r^(-3/4)</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'heavier hole = cooler disk (T_* ~ M^(-1/4)): X-ray binary vs UV quasar</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+24}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">log10 radius (r / r_in) -&gt;</text>')
    parts.append(f'<text x="{pad-14}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'log10 temperature (K)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
