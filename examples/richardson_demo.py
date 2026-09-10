"""Demo: the Richardson number -- shear versus stratification.

Prints the Richardson number and Kelvin-Helmholtz stability for atmospheric and ocean layers,
then draws the stability map over stratification and shear with the Ri = 1/4 threshold, plus
a sketch of the cat's-eye billows that grow when shear wins.

    python examples/richardson_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from richardson import (gradient_richardson, is_kh_stable, critical_shear,  # noqa: E402
                        brunt_vaisala_frequency, RI_CRITICAL)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Richardson number Ri = N^2 / (du/dz)^2: buoyant stiffness vs shear (KH at Ri<1/4)\n")
    print(f"  {'layer':<26}{'N (rad/s)':>11}{'shear':>10}{'Ri':>9}{'state':>14}")
    # (name, N, shear du/dz)
    layers = [
        ("ocean thermocline", 0.010, 0.005),
        ("nocturnal inversion", 0.020, 0.02),
        ("jet-stream shear zone", 0.012, 0.03),
        ("breaking billows", 0.008, 0.05),
    ]
    for name, N, shear in layers:
        Ri = gradient_richardson(N * N, shear)
        state = "layered (stable)" if is_kh_stable(Ri) else "KH billows"
        print(f"  {name:<26}{N:>11.3f}{shear:>10.3f}{Ri:>9.2f}{state:>14}")

    N = 0.012
    sc = critical_shear(N * N)
    print("\n  For N = %.3f rad/s the critical shear is %.4f /s: steeper than that drops Ri" % (N, sc))
    print("  below 1/4 and the interface rolls up into Kelvin-Helmholtz cat's-eye billows --")
    print("  the same physics behind clear-air turbulence that jolts aircraft and cloud-edge waves.")

    _svg(os.path.join(outdir, "richardson.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'richardson.svg')}")


def _svg(path, size=720, pad=80):
    # Top: stability map (x = stratification N, y = shear du/dz), Ri=1/4 boundary.
    # Bottom: sketch of KH billows.
    x0, x1 = pad, size - pad
    mid = size * 0.56

    # --- map ---
    N_max = 0.03
    S_max = 0.08
    nx, ny = 90, 70
    ty0, ty1 = mid - 24, pad + 44

    def X(N):
        return x0 + N / N_max * (x1 - x0)

    def Y(s):
        return ty0 - s / S_max * (ty0 - ty1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="34" fill="#e6edf3" font-size="18">Richardson stability map</text>',
        f'<text x="20" y="52" fill="#8b949e" font-size="12">'
        f'strong shear (low Ri) rips a stratified interface into Kelvin-Helmholtz billows</text>',
    ]

    cw = (x1 - x0) / nx
    ch = (ty0 - ty1) / ny
    for j in range(ny):
        s = S_max * (j + 0.5) / ny
        yy = Y(s) - ch
        for i in range(nx):
            N = N_max * (i + 0.5) / nx
            Ri = gradient_richardson(N * N, s) if s > 0 else 1e9
            col = "#4dabf7" if is_kh_stable(Ri) else "#ff6b6b"    # blue stable, red billows
            parts.append(f'<rect x="{X(N)-cw/2:.1f}" y="{yy:.1f}" width="{cw+0.8:.1f}" '
                         f'height="{ch+0.8:.1f}" fill="{col}" opacity="0.7"/>')

    # Ri = 1/4 boundary: s = N / sqrt(Ri_c) = 2N
    bpts = []
    for i in range(nx + 1):
        N = N_max * i / nx
        s = critical_shear(N * N)
        if s <= S_max:
            bpts.append(f"{X(N):.1f},{Y(s):.1f}")
    parts.append(f'<polyline points="{" ".join(bpts)}" fill="none" stroke="#ffd43b" '
                 f'stroke-width="2.6" stroke-dasharray="7 5"/>')
    parts.append(f'<text x="{X(0.02):.1f}" y="{Y(0.055):.1f}" fill="#ffd43b" font-size="11">Ri = 1/4</text>')

    parts.append(f'<rect x="{x0:.1f}" y="{ty1:.1f}" width="{x1-x0:.1f}" height="{ty0-ty1:.1f}" '
                 f'fill="none" stroke="#8b949e" stroke-width="1.5"/>')
    parts.append(f'<text x="{X(0.006):.1f}" y="{Y(0.01):.1f}" fill="#0d1117" font-size="12" '
                 f'font-weight="bold">stable (layered)</text>')
    parts.append(f'<text x="{X(0.004):.1f}" y="{Y(0.07):.1f}" fill="#0d1117" font-size="12" '
                 f'font-weight="bold">KH billows</text>')
    parts.append(f'<text x="{(x0+x1)/2:.1f}" y="{ty0+20:.1f}" fill="#8b949e" font-size="11" '
                 f'text-anchor="middle">stratification N (rad/s)</text>')
    parts.append(f'<text x="24" y="{(ty0+ty1)/2:.1f}" fill="#8b949e" font-size="11" '
                 f'transform="rotate(-90 24 {(ty0+ty1)/2:.1f})" text-anchor="middle">shear du/dz (1/s)</text>')

    # --- KH billow sketch ---
    by = size * 0.80
    bx0, bx1 = x0, x1
    amp = 26
    parts.append(f'<text x="{x0:.1f}" y="{mid+30:.1f}" fill="#8b949e" font-size="12">'
                 f'Kelvin-Helmholtz cat&#39;s-eye billows (the rolled-up interface):</text>')
    # rolled interface: a wavy line with curl
    n = 240
    pts = []
    for k in range(n + 1):
        t = k / n
        x = bx0 + t * (bx1 - bx0)
        phase = t * 4.0 * math.pi
        # billow: sine plus a curl that grows the crest
        y = by + amp * math.sin(phase) * (0.6 + 0.4 * math.sin(phase / 2))
        pts.append(f"{x:.1f},{y:.1f}")
    parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#06d6a0" stroke-width="2.6"/>')
    # fast layer arrow on top, slow on bottom
    parts.append(f'<line x1="{bx0+30:.1f}" y1="{by-amp-24:.1f}" x2="{bx0+140:.1f}" y2="{by-amp-24:.1f}" '
                 f'stroke="#8b949e" stroke-width="1.6"/>')
    parts.append(f'<polygon points="{bx0+140:.1f},{by-amp-24:.1f} {bx0+130:.1f},{by-amp-28:.1f} '
                 f'{bx0+130:.1f},{by-amp-20:.1f}" fill="#8b949e"/>')
    parts.append(f'<text x="{bx0+30:.1f}" y="{by-amp-30:.1f}" fill="#8b949e" font-size="10">fast light layer</text>')
    parts.append(f'<text x="{bx0+30:.1f}" y="{by+amp+34:.1f}" fill="#8b949e" font-size="10">slow dense layer</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
