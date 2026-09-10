"""Demo: the Tolman surface-brightness dimming test.

Prints the surface-brightness dimming with redshift for the expanding and tired-light
universes, then draws both curves in magnitudes so the (1+z)^4 expanding law visibly
diverges from the (1+z)^1 static prediction -- the test that confirms real expansion.

    python examples/tolman_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tolman import (dimming_factor, dimming_magnitudes,  # noqa: E402
                    tired_light_factor, expanding_vs_tired)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Tolman test: surface brightness ~ (1+z)^-4 if the universe expands\n")
    print(f"  {'z':>6}{'expanding':>14}{'mag':>8}{'tired-light':>14}"
          f"{'ratio E/T':>11}")
    print("  " + "-" * 53)
    for z in (0.2, 0.5, 1.0, 2.0, 3.0, 5.0):
        exp = dimming_factor(z)
        tired = tired_light_factor(z)
        print(f"  {z:>6.1f}{'1/'+format(1/exp,'.0f'):>14}{dimming_magnitudes(z):>8.2f}"
              f"{'1/'+format(1/tired,'.1f'):>14}{expanding_vs_tired(z):>11.3f}")

    print("\n  Surface brightness is distance-independent in a static Euclidean universe")
    print("  -- flux and angular area fall together. Expansion breaks that with four")
    print("  factors of (1+z): redshift, time dilation, and the D_A/D_L geometry. So a")
    print("  z=1 galaxy is dimmed 16x per square arcsecond, a z=3 galaxy 256x. A")
    print("  tired-light universe would dim only as (1+z); observations back the (1+z)^4,")
    print("  direct evidence the redshift is genuine expansion, not photons tiring out.")

    _svg(os.path.join(outdir, "tolman.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'tolman.svg')}")


def _svg(path, size=720, pad=72):
    zs = [0.05 * i for i in range(0, 101)]   # 0 .. 5
    exp_mag = [dimming_magnitudes(z, 4.0) for z in zs]
    tired_mag = [dimming_magnitudes(z, 1.0) for z in zs]
    xmin, xmax = zs[0], zs[-1]
    ymax = max(exp_mag) * 1.05
    ymin = 0.0

    def sx(x):
        return pad + (x - xmin) / (xmax - xmin) * (size - 2 * pad)

    def sy(y):
        return size - pad - (y - ymin) / (ymax - ymin) * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    ep = " ".join(f"{sx(zs[i]):.1f},{sy(exp_mag[i]):.1f}" for i in range(len(zs)))
    tp = " ".join(f"{sx(zs[i]):.1f},{sy(tired_mag[i]):.1f}" for i in range(len(zs)))
    parts.append(f'<polyline points="{ep}" fill="none" stroke="#8338ec" stroke-width="2.8"/>')
    parts.append(f'<polyline points="{tp}" fill="none" stroke="#ff922b" stroke-width="2.2" '
                 f'stroke-dasharray="6 4"/>')

    parts.append(f'<text x="{pad+10}" y="{pad+22}" fill="#8338ec" font-size="12">'
                 f'expanding: (1+z)^4 dimming</text>')
    parts.append(f'<text x="{pad+10}" y="{pad+38}" fill="#ff922b" font-size="12">'
                 f'tired light: (1+z)^1 (ruled out)</text>')

    # mark z=1 dimming
    z1x = sx(1.0)
    parts.append(f'<circle cx="{z1x:.1f}" cy="{sy(dimming_magnitudes(1.0)):.1f}" '
                 f'r="4" fill="#ffd43b"/>')
    parts.append(f'<text x="{z1x+8:.1f}" y="{sy(dimming_magnitudes(1.0)):.1f}" '
                 f'fill="#ffd43b" font-size="10">z=1: 3 mag (16x)</text>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'The Tolman test: surface-brightness dimming vs redshift</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+24}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">redshift z -&gt;</text>')
    parts.append(f'<text x="{pad-18}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'dimming (magnitudes / arcsec^2)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
