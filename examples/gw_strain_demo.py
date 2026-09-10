"""Demo: gravitational-wave strain -- the tiny number LIGO measures.

Computes GW150914's chirp mass, strain, and LIGO arm-length change, compares
binaries, and shows how strain falls with distance. Renders strain vs distance
to a log-log SVG.

    python examples/gw_strain_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gw_strain import (chirp_mass, strain, arm_length_change,  # noqa: E402
                       gw150914_strain, M_SUN, MPC, LIGO_ARM)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    h = gw150914_strain()
    print("Gravitational-wave strain: measuring a proton-width in kilometers\n")
    print(f"  GW150914 (36 + 29 M_sun, 410 Mpc, f_gw~150 Hz):")
    print(f"    chirp mass    = {chirp_mass(36*M_SUN, 29*M_SUN)/M_SUN:.1f} M_sun")
    print(f"    strain h      = {h:.2e}")
    print(f"    LIGO arm move = {arm_length_change(h):.2e} m "
          f"({arm_length_change(h)/8e-16*100:.1f}% of a proton width)\n")

    print(f"  {'binary':<28}{'distance':>10}{'strain h':>12}")
    print("  " + "-" * 50)
    cases = [
        ("GW150914 (36+29)", 410, 36, 29),
        ("neutron stars (1.4+1.4)", 130, 1.4, 1.4),
        ("supermassive (1e6+1e6)", 1e4, 1e6, 1e6),
    ]
    for name, d_mpc, m1, m2 in cases:
        hh = strain(m1 * M_SUN, m2 * M_SUN, d_mpc * MPC, 150.0)
        print(f"  {name:<28}{d_mpc:>7.0f} Mpc{hh:>12.1e}")
    print("\n  A strain of 1e-21 moves LIGO's 4 km arms by ~1e-18 m -- a thousandth")
    print("  of a proton's width. Detecting it is why LIGO is one of the most")
    print("  sensitive instruments ever built.")

    _svg(os.path.join(outdir, "gw_strain.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'gw_strain.svg')}")


def _svg(path, size=720, pad=64):
    # strain vs distance for GW150914-type source (log-log)
    ds = [10 ** (1 + 0.03 * i) for i in range(0, 100)]  # 10 Mpc .. 1e4 Mpc
    hs = [strain(36 * M_SUN, 29 * M_SUN, d * MPC, 150.0) for d in ds]
    ld = [math.log10(d) for d in ds]
    lh = [math.log10(x) for x in hs]
    dmin, dmax = ld[0], ld[-1]
    hmin, hmax = min(lh), max(lh)

    def sx(x):
        return pad + (x - dmin) / (dmax - dmin) * (size - 2 * pad)

    def sy(y):
        return size - pad - (y - hmin) / (hmax - hmin) * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    poly = " ".join(f"{sx(ld[i]):.1f},{sy(lh[i]):.1f}" for i in range(len(ds)))
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#4cc9f0" stroke-width="2.2"/>')
    # mark GW150914 at 410 Mpc
    l410 = math.log10(410.0)
    parts.append(f'<line x1="{sx(l410):.1f}" y1="{pad}" x2="{sx(l410):.1f}" y2="{size-pad}" '
                 f'stroke="#e63946" stroke-dasharray="4,4"/>')
    parts.append(f'<text x="{sx(l410)+6:.1f}" y="{pad+16}" fill="#e63946" font-size="12">'
                 f'GW150914 (410 Mpc, h~2e-21)</text>')
    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'GW strain vs distance (36+29 M_sun binary)</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+22}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">log10 distance (Mpc) -&gt;</text>')
    parts.append(f'<text x="{pad-10}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'log10 strain h</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
