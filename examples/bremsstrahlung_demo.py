"""Demo: thermal bremsstrahlung -- the X-ray glow and cooling of cluster gas.

Shows the free-free emissivity and cooling time across cluster densities, marking
where the gas cools within a Hubble time (cooling flows). Renders cooling time
vs electron density to a log-log SVG.

    python examples/bremsstrahlung_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bremsstrahlung import (emissivity, cooling_time_gyr,  # noqa: E402
                            cools_within_hubble)

T = 5e7  # ~5 keV cluster gas


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print(f"Thermal bremsstrahlung of cluster gas (T = {T:.0e} K)\n")
    print(f"  {'n_e (/m^3)':>12}{'emissivity (W/m^3)':>20}{'t_cool (Gyr)':>14}{'flow?':>8}")
    print("  " + "-" * 54)
    for ne in (1e2, 1e3, 1e4, 1e5):
        flow = "yes" if cools_within_hubble(ne, T) else "no"
        print(f"  {ne:>12.0e}{emissivity(ne, T):>20.2e}{cooling_time_gyr(ne, T):>14.1f}{flow:>8}")
    print("\n  Emissivity goes as n^2, so dense cores glow brightest and cool fastest")
    print("  (t_cool ~ sqrt(T)/n). A cooling flow develops where t_cool drops below")
    print("  the Hubble time; the tenuous outskirts effectively never cool. This is")
    print("  the X-ray emission whose CMB imprint is the SZ effect.")

    _svg(outdir + "/bremsstrahlung.svg")
    print(f"\n  wrote {outdir}/bremsstrahlung.svg")


def _svg(path, size=720, pad=64):
    nes = [10 ** (1 + 0.05 * i) for i in range(0, 101)]  # 10 .. 1e6 /m^3
    tc = [cooling_time_gyr(ne, T) for ne in nes]
    ln = [math.log10(ne) for ne in nes]
    lt = [math.log10(t) for t in tc]
    nmin, nmax = ln[0], ln[-1]
    tmin, tmax = min(lt), max(lt)

    def sx(x):
        return pad + (x - nmin) / (nmax - nmin) * (size - 2 * pad)

    def sy(y):
        return size - pad - (y - tmin) / (tmax - tmin) * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    # Hubble time line
    lh = math.log10(13.8)
    if tmin < lh < tmax:
        parts.append(f'<line x1="{pad}" y1="{sy(lh):.1f}" x2="{size-pad}" y2="{sy(lh):.1f}" '
                     f'stroke="#e63946" stroke-dasharray="5,4"/>')
        parts.append(f'<text x="{size-pad-4}" y="{sy(lh)-6:.1f}" fill="#e63946" font-size="12" '
                     f'text-anchor="end">Hubble time (cooling-flow threshold)</text>')
    poly = " ".join(f"{sx(ln[i]):.1f},{sy(lt[i]):.1f}" for i in range(len(nes)))
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#4cc9f0" stroke-width="2.2"/>')
    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Bremsstrahlung cooling time vs density</text>')
    parts.append(f'<text x="{pad}" y="{size-pad+24}" fill="#8b949e" font-size="11">'
                 f'log10 electron density (/m^3); below the line = cooling flow</text>')
    parts.append(f'<text x="{pad-10}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'log10 cooling time (Gyr)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
