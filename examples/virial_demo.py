"""Demo: the virial theorem and violent relaxation of a star cluster.

Integrates (a) an equilibrium Plummer sphere and (b) a COLD, sub-virial cluster.
The equilibrium one hovers at the virial value 2T/U = -1; the cold one collapses,
overshoots, and relaxes toward -1. Prints the running virial ratio as a sparkline
and renders both tracks to SVG.

    python examples/virial_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from virial import run_virial, virial_ratio, cold_collapse  # noqa: E402
from systems import plummer_sphere  # noqa: E402

_BARS = " .:-=+*#@"


def sparkline(vals, lo=-1.4, hi=0.1):
    span = (hi - lo) or 1.0
    return "".join(_BARS[max(0, min(8, int((v - lo) / span * 8)))] for v in vals)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    n, dt, steps = 200, 0.02, 4000
    print("Virial theorem: a bound gravitational system settles at 2T/U = -1\n")

    eq = plummer_sphere(n=n, seed=3)
    ts_e, r_e, run_e = run_virial(eq, dt, steps, sample_every=20)

    cold = cold_collapse(n=n, seed=3, coldness=0.3)
    start_cold = virial_ratio(cold)
    ts_c, r_c, run_c = run_virial(cold, dt, steps, sample_every=20)

    print(f"  equilibrium Plummer : running <2T/U> -> {run_e[-1]:+.3f}  (target -1)")
    print(f"  cold cluster        : start {start_cold:+.3f} -> running <2T/U> "
          f"{run_c[-1]:+.3f}\n")

    stride = max(1, len(r_c) // 70)
    print("  cold-cluster instantaneous 2T/U (collapse & relaxation):")
    print("  " + sparkline(r_c[::stride]))

    # SVG of both running averages vs time
    _svg([(ts_e, run_e, "#2a9d8f", "equilibrium"),
          (ts_c, run_c, "#e63946", "cold collapse")],
         os.path.join(outdir, "virial.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'virial.svg')}")
    print("  Both running averages converge on -1: the cold cluster forgets its")
    print("  cold start through violent relaxation and lands in virial balance.")


def _svg(tracks, path, size=720, pad=56):
    all_t = [t for ts, _, _, _ in tracks for t in ts]
    tmax = max(all_t)

    def sx(t):
        return pad + t / tmax * (size - 2 * pad)

    def sy(v):  # map [-1.4, 0.1] to plot
        lo, hi = -1.4, 0.1
        return size - (pad + (v - lo) / (hi - lo) * (size - 2 * pad))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        # virial target line at -1
        f'<line x1="{pad}" y1="{sy(-1):.1f}" x2="{size-pad}" y2="{sy(-1):.1f}" '
        f'stroke="#e9c46a" stroke-dasharray="5,4" stroke-width="1"/>',
        f'<text x="{size-pad-4}" y="{sy(-1)-6:.1f}" fill="#e9c46a" font-size="11" '
        f'text-anchor="end">2T/U = -1 (virial)</text>',
    ]
    for ts, vals, col, label in tracks:
        poly = " ".join(f"{sx(ts[i]):.1f},{sy(vals[i]):.1f}" for i in range(len(ts)))
        parts.append(f'<polyline points="{poly}" fill="none" stroke="{col}" '
                     f'stroke-width="1.8"/>')
    parts.append(f'<text x="{pad}" y="30" fill="#e6edf3" font-size="18">'
                 f'Virial theorem &amp; violent relaxation</text>')
    parts.append(f'<text x="{pad}" y="50" fill="#8b949e" font-size="12">'
                 f'running &lt;2T/U&gt; vs time; green=equilibrium, red=cold cluster</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
