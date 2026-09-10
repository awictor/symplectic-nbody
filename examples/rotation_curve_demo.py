"""Demo: galaxy rotation curves and the case for dark matter.

Builds the circular-speed curve for a visible disk alone (Keplerian decline) and
for disk + NFW dark halo (flat), prints the outer log-log slopes, and renders
both curves to SVG. The gap between them is the dark matter.

    python examples/rotation_curve_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rotation_curve import rotation_curve, keplerian_tail_slope  # noqa: E402

_BARS = " .:-=+*#@"


def spark(vals, lo, hi):
    span = (hi - lo) or 1.0
    return "".join(_BARS[max(0, min(8, int((v - lo) / span * 8)))] for v in vals)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    radii = [0.1 + 0.2 * i for i in range(60)]
    v_vis, v_tot = rotation_curve(radii, M_disk=1.0, R_d=1.0, halo=(0.02, 5.0), G=1.0)

    print("Galaxy rotation curves: the case for dark matter\n")
    print(f"  visible disk only : outer slope {keplerian_tail_slope(radii, v_vis):+.3f} "
          f"(Keplerian decline = -0.50)")
    print(f"  disk + dark halo  : outer slope {keplerian_tail_slope(radii, v_tot):+.3f} "
          f"(flat = 0.00)\n")

    hi = max(max(v_vis), max(v_tot))
    print("  visible-only v(r):", spark(v_vis, 0, hi))
    print("  disk+halo  v(r):", spark(v_tot, 0, hi))
    print("\n  The visible curve falls off; the observed curve is flat. The gap is")
    print("  the dark-matter halo, whose enclosed mass keeps growing as ~ r.")

    _svg(radii, v_vis, v_tot, os.path.join(outdir, "rotation_curve.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'rotation_curve.svg')}")


def _svg(radii, v_vis, v_tot, path, size=720, pad=60):
    rmax = radii[-1]
    vmax = max(max(v_vis), max(v_tot)) * 1.1

    def sx(r):
        return pad + r / rmax * (size - 2 * pad)

    def sy(v):
        return size - pad - v / vmax * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    for vals, col, label in ((v_tot, "#4cc9f0", "disk + dark halo (observed, flat)"),
                             (v_vis, "#e63946", "visible disk only (Keplerian)")):
        poly = " ".join(f"{sx(radii[i]):.1f},{sy(vals[i]):.1f}" for i in range(len(radii)))
        parts.append(f'<polyline points="{poly}" fill="none" stroke="{col}" stroke-width="2"/>')
    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Galaxy rotation curve: dark matter keeps it flat</text>')
    parts.append(f'<text x="{pad+14}" y="{pad+8}" fill="#4cc9f0" font-size="12">'
                 f'disk + dark halo (flat)</text>')
    parts.append(f'<text x="{pad+14}" y="{pad+26}" fill="#e63946" font-size="12">'
                 f'visible disk only (declines)</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+22}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">radius -&gt;</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
