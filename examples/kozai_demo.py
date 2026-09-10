"""Demo: Kozai-Lidov eccentricity <-> inclination cycles in a hierarchical triple.

Integrates the secular flow for a near-circular inner orbit at high inclination
and shows the eccentricity and inclination oscillating out of phase (trading the
conserved Theta). Compares e_max to the analytic prediction, and renders the
cycles as time series plus the (e, i) phase track to SVG.

    python examples/kozai_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kozai import evolve, analytic_emax, CRITICAL_ANGLE_DEG  # noqa: E402

_BARS = " .:-=+*#@"


def spark(vals, lo, hi):
    span = (hi - lo) or 1.0
    return "".join(_BARS[max(0, min(8, int((v - lo) / span * 8)))] for v in vals)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    i0 = 80.0
    taus, es, idegs, Fs = evolve(0.01, i0, omega0_deg=90.0,
                                 dtau=1e-3, n_steps=180000, sample_every=300)
    print("Kozai-Lidov cycles in a hierarchical triple\n")
    print(f"  critical inclination      : {CRITICAL_ANGLE_DEG:.2f} deg")
    print(f"  start                     : e=0.01, i={i0:.0f} deg")
    print(f"  e_max measured / analytic : {max(es):.3f} / {analytic_emax(i0):.3f}")
    print(f"  inclination swings        : {min(idegs):.1f} - {max(idegs):.1f} deg\n")

    stride = max(1, len(es) // 70)
    print("  eccentricity :", spark(es[::stride], 0.0, 1.0))
    print("  inclination  :", spark(idegs[::stride], min(idegs), max(idegs)))
    print("\n  e peaks exactly when i dips: they trade the conserved")
    print("  Theta = sqrt(1-e^2) cos i. This drives hot-Jupiter migration")
    print("  and compact-binary mergers.")

    _svg(taus, es, idegs, os.path.join(outdir, "kozai.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'kozai.svg')}")


def _svg(taus, es, idegs, path, size=720, pad=56):
    tmax = taus[-1]
    imin, imax = min(idegs), max(idegs)

    def sx(t):
        return pad + t / tmax * (size - 2 * pad)

    def sy_e(e):  # eccentricity 0..1 on left half scale
        return size - (pad + e * (size - 2 * pad))

    def sy_i(i):  # inclination mapped to same box
        return size - (pad + (i - imin) / ((imax - imin) or 1) * (size - 2 * pad))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
    ]
    e_poly = " ".join(f"{sx(taus[k]):.1f},{sy_e(es[k]):.1f}" for k in range(len(taus)))
    i_poly = " ".join(f"{sx(taus[k]):.1f},{sy_i(idegs[k]):.1f}" for k in range(len(taus)))
    parts.append(f'<polyline points="{e_poly}" fill="none" stroke="#e63946" stroke-width="1.8"/>')
    parts.append(f'<polyline points="{i_poly}" fill="none" stroke="#4cc9f0" stroke-width="1.8"/>')
    parts.append(f'<text x="{pad}" y="30" fill="#e6edf3" font-size="18">'
                 f'Kozai-Lidov cycles: e (red) and inclination (blue)</text>')
    parts.append(f'<text x="{pad}" y="50" fill="#8b949e" font-size="12">'
                 f'out of phase -- when e rises, i falls; Theta=sqrt(1-e^2)cos i is fixed</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
