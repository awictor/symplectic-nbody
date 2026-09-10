"""Demo: the magnetic mirror and the loss cone.

Prints the loss-cone angle and a sample of trapped/lost pitch angles for several
mirror ratios, then draws the loss-cone angle vs mirror ratio, shading the trapped
region -- the geometry that holds the Van Allen belts and mirror-machine plasmas.

    python examples/magnetic_mirror_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from magnetic_mirror import (loss_cone_angle, is_trapped,  # noqa: E402
                             mirror_field, mirror_ratio)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Magnetic mirror: trapped if sin^2(pitch) > B_min/B_max = 1/R_m\n")
    print(f"  {'mirror ratio R_m':>18}{'loss cone (deg)':>18}"
          f"{'20 deg':>9}{'60 deg':>9}")
    print("  " + "-" * 54)
    for R in (2.0, 4.0, 10.0, 50.0, 1000.0):
        a_lc = math.degrees(loss_cone_angle(R, 1.0))
        t20 = "trap" if is_trapped(math.radians(20), R, 1.0) else "lost"
        t60 = "trap" if is_trapped(math.radians(60), R, 1.0) else "lost"
        print(f"  {R:>18.0f}{a_lc:>18.1f}{t20:>9}{t60:>9}")

    print("\n  A bigger mirror ratio means a narrower loss cone, so more particles are")
    print("  held. Earth's dipole (R_m ~ tens between equator and pole) traps the Van")
    print("  Allen belts: particles bounce pole to pole, reflected where the field")
    print("  tightens, unless their pitch angle drops them into the loss cone and down")
    print("  into the atmosphere -- which is what lights up the aurora.")

    _svg(os.path.join(outdir, "magnetic_mirror.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'magnetic_mirror.svg')}")


def _svg(path, size=720, pad=70):
    Rs = [10 ** (0.05 * i) for i in range(1, 61)]   # ~1.1 .. 1000
    angs = [math.degrees(loss_cone_angle(R, 1.0)) for R in Rs]
    lx = [math.log10(R) for R in Rs]
    xmin, xmax = lx[0], lx[-1]
    ymin, ymax = 0.0, 90.0

    def sx(x):
        return pad + (x - xmin) / (xmax - xmin) * (size - 2 * pad)

    def sy(y):
        return size - pad - (y - ymin) / (ymax - ymin) * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
    ]
    # trapped region: above the loss-cone curve (shade the upper area)
    top = [f"{sx(lx[i]):.1f},{sy(angs[i]):.1f}" for i in range(len(Rs))]
    top_poly = (f"{sx(xmax):.1f},{sy(90):.1f} {sx(xmin):.1f},{sy(90):.1f} "
                + " ".join(top))
    parts.append(f'<polygon points="{top_poly}" fill="#06d6a0" fill-opacity="0.10"/>')

    parts.append(f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>')
    parts.append(f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>')

    poly = " ".join(top)
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#ff6b6b" stroke-width="2.6"/>')

    parts.append(f'<text x="{sx(math.log10(3)):.1f}" y="{sy(65):.1f}" fill="#06d6a0" '
                 f'font-size="13">trapped (mirrors)</text>')
    parts.append(f'<text x="{sx(math.log10(20)):.1f}" y="{sy(12):.1f}" fill="#ff6b6b" '
                 f'font-size="13">loss cone (escapes)</text>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'The loss cone shrinks as the mirror tightens</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'loss-cone angle = arcsin(sqrt(1/R_m)); above it particles are held</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+24}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">log10 mirror ratio R_m -&gt;</text>')
    parts.append(f'<text x="{pad-16}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'equatorial pitch angle (deg)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
