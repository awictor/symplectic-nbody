"""Demo: the Sitnikov problem's transition from order to chaos.

Builds stroboscopic Poincare maps (sample (z, vz) once per binary period) for a
circular binary and an eccentric one. The circular case draws smooth nested
curves; the eccentric case shreds them into a chaotic layer -- Moser's theorem
made visible.

    python examples/sitnikov_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sitnikov import poincare_map  # noqa: E402

_PALETTE = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a", "#f4a261",
            "#8338ec", "#3a86ff", "#ff006e", "#06d6a0", "#ffbe0b",
            "#b5179e", "#4cc9f0", "#fb8500", "#90be6d"]


def _map_svg(orbits, path, title, sub, size=680, pad=44):
    all_z = [p[0] for pts in orbits for p in pts]
    all_v = [p[1] for pts in orbits for p in pts]
    if not all_z:
        all_z, all_v = [0.0], [0.0]
    zmin, zmax = min(all_z), max(all_z)
    vmin, vmax = min(all_v), max(all_v)

    def sx(z):
        return pad + (z - zmin) / ((zmax - zmin) or 1) * (size - 2 * pad)

    def sy(v):
        return size - (pad + (v - vmin) / ((vmax - vmin) or 1) * (size - 2 * pad))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
    ]
    for k, pts in enumerate(orbits):
        col = _PALETTE[k % len(_PALETTE)]
        dots = "".join(f'<circle cx="{sx(z):.1f}" cy="{sy(v):.1f}" r="1.2" '
                       f'fill="{col}"/>' for z, v in pts)
        parts.append(dots)
    parts.append(f'<text x="{pad}" y="28" fill="#e6edf3" font-size="17">{title}</text>')
    parts.append(f'<text x="{pad}" y="{size-14}" fill="#8b949e" font-size="11">{sub}</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    seeds = [0.15, 0.3, 0.45, 0.6, 0.8, 1.0, 1.2, 1.4, 1.7, 2.0]
    print("Sitnikov problem: stroboscopic Poincare maps (sample once per binary period)\n")

    for e, tag, title in [
        (0.0, "circ", "Sitnikov: circular binary (e=0) -- integrable"),
        (0.3, "ecc", "Sitnikov: eccentric binary (e=0.3) -- chaotic layer"),
    ]:
        orbits = []
        for z0 in seeds:
            pts = poincare_map(z0, 0.0, e, n_periods=250, steps_per_period=1500)
            if len(pts) >= 5:
                orbits.append(pts)
        total = sum(len(o) for o in orbits)
        # a rough order/chaos read: mean per-orbit z-spread
        spreads = [max(p[0] for p in o) - min(p[0] for p in o) for o in orbits]
        print(f"  e={e}: {len(orbits)} orbits survived, {total} section points, "
              f"mean z-spread {sum(spreads)/len(spreads):.2f}")
        path = os.path.join(outdir, f"sitnikov_{tag}.svg")
        _map_svg(orbits, path, title,
                 "axes (z, vz) sampled once per binary period; hollow curves = tori, dust = chaos")
        print(f"  wrote {path}")

    print("\ne=0 draws smooth nested curves (each orbit lies on an invariant torus).")
    print("e=0.3 tears the inner curves into a chaotic sea -- the Sitnikov route to chaos.")


if __name__ == "__main__":
    main()
