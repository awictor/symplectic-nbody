"""Demo: galaxy-cluster virial temperature and the mass-temperature relation.

Computes the virial temperature of Coma-like clusters, shows the kT ~ M^{2/3}
mass-temperature relation used to weigh clusters from X-ray data, and renders it
to a log-log SVG.

    python examples/cluster_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cluster import kT_kev, virial_temperature, M_SUN, MPC  # noqa: E402


def radius_for_mass(M):
    """Virial radius at fixed overdensity: R ~ M^{1/3}, normalized to 2 Mpc at
    1e15 M_sun."""
    return 2.0 * MPC * (M / (1e15 * M_SUN)) ** (1.0 / 3.0)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Galaxy clusters: virial temperature and the M-T relation\n")
    print(f"  {'mass (M_sun)':>14}{'radius (Mpc)':>14}{'kT (keV)':>10}{'T (K)':>12}")
    print("  " + "-" * 50)
    for Mexp in (13, 13.5, 14, 14.5, 15):
        M = 10 ** Mexp * M_SUN
        R = radius_for_mass(M)
        print(f"  {10**Mexp:>14.0e}{R/MPC:>14.2f}{kT_kev(M, R):>10.2f}"
              f"{virial_temperature(M, R):>12.1e}")
    print("\n  A massive cluster reaches several keV (~10^8 K), radiating thermal")
    print("  bremsstrahlung X-rays. Because kT ~ M^{2/3}, an X-ray temperature")
    print("  measures the cluster's total (mostly dark) mass -- how clusters are weighed.")

    _svg(os.path.join(outdir, "cluster.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'cluster.svg')}")


def _svg(path, size=720, pad=64):
    masses = [10 ** (13 + 0.02 * i) * M_SUN for i in range(0, 101)]  # 1e13..1e15
    kts = [kT_kev(M, radius_for_mass(M)) for M in masses]
    lm = [math.log10(M / M_SUN) for M in masses]
    lk = [math.log10(k) for k in kts]
    mmin, mmax = lm[0], lm[-1]
    kmin, kmax = min(lk), max(lk)

    def sx(x):
        return pad + (x - mmin) / (mmax - mmin) * (size - 2 * pad)

    def sy(y):
        return size - pad - (y - kmin) / (kmax - kmin) * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    poly = " ".join(f"{sx(lm[i]):.1f},{sy(lk[i]):.1f}" for i in range(len(masses)))
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#f4a261" stroke-width="2.2"/>')
    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Cluster mass-temperature relation (kT ~ M^2/3)</text>')
    parts.append(f'<text x="{pad}" y="{size-pad+24}" fill="#8b949e" font-size="11">'
                 f'log10 cluster mass (M_sun) -&gt;</text>')
    parts.append(f'<text x="{pad-10}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'log10 kT (keV)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
