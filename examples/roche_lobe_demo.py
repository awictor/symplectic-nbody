"""Demo: Roche lobes and binary mass transfer.

Shows the Eggleton Roche-lobe radius vs mass ratio and the stability boundary of
conservative mass transfer, and renders R_L/a vs q to SVG.

    python examples/roche_lobe_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from roche_lobe import (eggleton_radius, l1_distance,  # noqa: E402
                        conservative_orbit_response, transfer_is_stable)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Roche lobes and binary mass transfer\n")
    print(f"  {'q = M_donor/M_acc':>18}{'R_L/a':>9}{'d ln a/d ln M':>16}{'transfer':>12}")
    print("  " + "-" * 55)
    for q in (0.2, 0.5, 1.0, 2.0, 5.0):
        stab = "stable" if transfer_is_stable(q) else "unstable"
        print(f"  {q:>18.1f}{eggleton_radius(q):>9.3f}"
              f"{conservative_orbit_response(q):>16.1f}{stab:>12}")
    print("\n  When a star fills its Roche lobe, gas pours through L1 onto its")
    print("  companion. From a lighter donor the orbit widens and transfer is stable")
    print("  (cataclysmic variables, X-ray binaries); from a heavier donor it runs")
    print("  away -- the path to mergers and type-Ia supernovae.")

    _svg(os.path.join(outdir, "roche_lobe.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'roche_lobe.svg')}")


def _svg(path, size=720, pad=64):
    qs = [10 ** (-1 + 0.02 * i) for i in range(0, 101)]   # 0.1 .. 10
    rl = [eggleton_radius(q) for q in qs]
    lq = [math.log10(q) for q in qs]
    qmin, qmax = lq[0], lq[-1]
    rmax = max(rl) * 1.1

    def sx(x):
        return pad + (x - qmin) / (qmax - qmin) * (size - 2 * pad)

    def sy(r):
        return size - pad - r / rmax * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    # stability boundary at q=1
    parts.append(f'<line x1="{sx(0):.1f}" y1="{pad}" x2="{sx(0):.1f}" y2="{size-pad}" '
                 f'stroke="#e63946" stroke-dasharray="4,4"/>')
    parts.append(f'<text x="{sx(0)+6:.1f}" y="{pad+16}" fill="#e63946" font-size="12">'
                 f'q=1: stability boundary</text>')
    poly = " ".join(f"{sx(lq[i]):.1f},{sy(rl[i]):.1f}" for i in range(len(qs)))
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#4cc9f0" stroke-width="2.2"/>')
    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Roche-lobe radius vs mass ratio (Eggleton)</text>')
    parts.append(f'<text x="{pad}" y="{size-pad+24}" fill="#8b949e" font-size="11">'
                 f'log10 q = M_donor/M_accretor; left of q=1 = stable transfer</text>')
    parts.append(f'<text x="{pad-10}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'R_L / a</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
