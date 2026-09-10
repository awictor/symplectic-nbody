"""Demo: optical depth and radiative transfer -- where a star's surface is.

Shows the exp(-tau) transmission law, the thin/thick transition, and the
photospheric tau = 2/3 that defines a star's visible surface. Renders
transmitted fraction vs optical depth to SVG.

    python examples/optical_depth_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from optical_depth import (transmitted_fraction, mean_free_path,  # noqa: E402
                           is_optically_thick, photosphere_depth, PHOTOSPHERE_TAU)

SIGMA_T = 6.6525e-29


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Optical depth: how far light sees into matter\n")
    print(f"  {'tau':>6}{'transmitted':>14}{'regime':>16}")
    print("  " + "-" * 38)
    for tau in (0.1, 0.5, PHOTOSPHERE_TAU, 1.0, 3.0, 10.0):
        reg = "thick" if is_optically_thick(tau) else "thin"
        note = " (photosphere)" if abs(tau - PHOTOSPHERE_TAU) < 1e-6 else ""
        print(f"  {tau:>6.2f}{transmitted_fraction(tau):>14.3f}{reg + note:>16}")
    n = 1e20
    print(f"\n  at n_e = {n:.0e} /m^3 (Thomson): mean free path {mean_free_path(n, SIGMA_T)/1e3:.0f} km,")
    print(f"  photosphere depth {photosphere_depth(n, SIGMA_T)/1e3:.0f} km.\n")
    print("  A star has no solid surface -- its photosphere is simply the layer")
    print("  where the inward optical depth reaches ~2/3, the depth from which")
    print("  photons finally escape and set the effective temperature.")

    _svg(os.path.join(outdir, "optical_depth.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'optical_depth.svg')}")


def _svg(path, size=720, pad=64):
    taus = [0.05 * i for i in range(0, 101)]  # 0 .. 5
    trans = [transmitted_fraction(t) for t in taus]

    def sx(t):
        return pad + t / 5.0 * (size - 2 * pad)

    def sy(f):
        return size - pad - f * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    poly = " ".join(f"{sx(taus[i]):.1f},{sy(trans[i]):.1f}" for i in range(len(taus)))
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#4cc9f0" stroke-width="2.2"/>')
    # photosphere line at tau = 2/3
    parts.append(f'<line x1="{sx(PHOTOSPHERE_TAU):.1f}" y1="{pad}" '
                 f'x2="{sx(PHOTOSPHERE_TAU):.1f}" y2="{size-pad}" '
                 f'stroke="#e9c46a" stroke-dasharray="4,4"/>')
    parts.append(f'<text x="{sx(PHOTOSPHERE_TAU)+6:.1f}" y="{pad+16}" fill="#e9c46a" '
                 f'font-size="12">photosphere (tau=2/3)</text>')
    # tau=1 marker
    parts.append(f'<line x1="{sx(1.0):.1f}" y1="{pad}" x2="{sx(1.0):.1f}" y2="{size-pad}" '
                 f'stroke="#30363d" stroke-dasharray="2,4"/>')
    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Radiative transfer: transmitted fraction e^-tau</text>')
    parts.append(f'<text x="{pad}" y="{size-pad+24}" fill="#8b949e" font-size="11">'
                 f'optical depth tau; thin (see through) to thick (see only surface)</text>')
    parts.append(f'<text x="{pad-10}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'transmitted fraction</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
