"""Demo: the relativistic Doppler effect.

Prints the receding, approaching and transverse shifts across speeds, then draws the
redshift z vs beta for the receding-radial and transverse (time-dilation) cases,
showing how the transverse redshift -- pure time dilation -- has no classical analogue.

    python examples/relativistic_doppler_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from relativistic_doppler import (lorentz_factor, redshift_radial,  # noqa: E402
                                  transverse_redshift, frequency_ratio_radial,
                                  transverse_frequency_ratio, beta_from_redshift)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Relativistic Doppler: classical shift x time dilation\n")
    print(f"  {'beta':>7}{'receding z':>13}{'approaching z':>15}"
          f"{'transverse z':>15}")
    print("  " + "-" * 50)
    for beta in (0.1, 0.3, 0.5, 0.7, 0.9, 0.99):
        print(f"  {beta:>7.2f}{redshift_radial(beta):>13.4f}"
              f"{redshift_radial(beta, approaching=True):>15.4f}"
              f"{transverse_redshift(beta):>15.4f}")

    print("\n  Receding sources redshift, approaching ones blueshift -- the familiar")
    print("  Doppler shift, but amplified by time dilation. The purely relativistic")
    print("  effect is the transverse shift: a source moving exactly across the line of")
    print("  sight still reddens by 1/gamma because its clock runs slow. Ives and")
    print("  Stilwell measured it in 1938 -- direct proof of time dilation. A measured")
    print(f"  redshift maps back to a speed: z = 1 means beta = {beta_from_redshift(1.0):.2f}.")

    _svg(os.path.join(outdir, "relativistic_doppler.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'relativistic_doppler.svg')}")


def _svg(path, size=720, pad=72):
    betas = [0.01 * i for i in range(1, 100)]   # 0.01 .. 0.99
    z_rec = [redshift_radial(b) for b in betas]
    z_app = [redshift_radial(b, approaching=True) for b in betas]  # negative
    z_tr = [transverse_redshift(b) for b in betas]
    xmin, xmax = 0.0, 1.0
    allv = z_rec + z_app + z_tr
    ymin, ymax = min(allv), max(allv)
    # clamp huge receding z for display
    ymax = min(ymax, 5.0)

    def sx(x):
        return pad + (x - xmin) / (xmax - xmin) * (size - 2 * pad)

    def sy(y):
        return size - pad - (y - ymin) / (ymax - ymin) * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
    ]
    # z = 0 axis
    y0 = sy(0.0)
    parts.append(f'<line x1="{pad}" y1="{y0:.1f}" x2="{size-pad}" y2="{y0:.1f}" stroke="#30363d"/>')
    parts.append(f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>')
    parts.append(f'<text x="{pad+6:.1f}" y="{y0-5:.1f}" fill="#8b949e" font-size="10">z = 0</text>')

    def poly(zs, col, w=2.4):
        pts = " ".join(f"{sx(betas[i]):.1f},{sy(min(zs[i], ymax)):.1f}"
                       for i in range(len(betas)))
        return f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="{w}"/>'
    parts.append(poly(z_rec, "#ff6b6b", 2.6))
    parts.append(poly(z_tr, "#ffd43b"))
    parts.append(poly(z_app, "#4dabf7"))

    parts.append(f'<text x="{pad+10}" y="{pad+22}" fill="#ff6b6b" font-size="12">receding (redshift)</text>')
    parts.append(f'<text x="{pad+10}" y="{pad+38}" fill="#ffd43b" font-size="12">transverse (1/gamma, pure time dilation)</text>')
    parts.append(f'<text x="{pad+10}" y="{pad+54}" fill="#4dabf7" font-size="12">approaching (blueshift, z&lt;0)</text>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Relativistic Doppler redshift vs speed</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+24}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">beta = v/c -&gt;</text>')
    parts.append(f'<text x="{pad-18}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'redshift z</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
