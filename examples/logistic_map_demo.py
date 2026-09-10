"""Demo: the logistic map -- period doubling into chaos.

Prints the attractor period and Lyapunov exponent as the growth rate r rises through the
period-doubling cascade, then draws the famous bifurcation diagram (attractor points vs r)
above the Lyapunov exponent, which dips to zero at each bifurcation and turns positive in the
chaotic regime.

    python examples/logistic_map_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from logistic_map import (step, attractor, period, lyapunov_exponent,  # noqa: E402
                          FEIGENBAUM_DELTA, CHAOS_ONSET)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Logistic map x' = r x (1-x): the route to chaos (Feigenbaum delta = %.4f)\n"
          % FEIGENBAUM_DELTA)
    print(f"  {'r':>8}{'period':>10}{'Lyapunov':>12}{'regime':>14}")
    for r in (2.5, 3.2, 3.5, 3.55, 3.83, 3.9, 4.0):
        p = period(r)
        lam = lyapunov_exponent(r)
        regime = "chaos" if lam > 0 else ("period-%d" % p if p <= 8 else "cycle")
        print(f"  {r:>8.2f}{p:>10}{lam:>12.3f}{regime:>14}")

    print("\n  Period doublings 2,4,8,... accumulate at r ~ %.4f, then chaos -- broken by" % CHAOS_ONSET)
    print("  periodic windows (the period-3 near 3.83 is the most famous). The bifurcation")
    print("  spacings shrink by the universal Feigenbaum ratio 4.669, the same for any smooth")
    print("  unimodal map -- one of the deepest facts in nonlinear dynamics.")

    _svg(os.path.join(outdir, "logistic_map.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'logistic_map.svg')}")


def _svg(path, size=760, pad=64):
    r_min, r_max = 2.8, 4.0
    x0, x1 = pad, size - pad
    # top: bifurcation diagram; bottom: Lyapunov exponent
    split = size * 0.66
    by0, by1 = split - 20, pad + 30      # bifurcation vertical band (x=0..1)

    def RX(r):
        return x0 + (r - r_min) / (r_max - r_min) * (x1 - x0)

    def XY(x):
        return by0 - x * (by0 - by1)     # attractor value 0..1

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="30" fill="#e6edf3" font-size="18">The logistic-map bifurcation diagram</text>',
        f'<text x="20" y="48" fill="#8b949e" font-size="12">'
        f'attractor vs growth rate: one point, then 2, 4, 8, ... into chaos (Lyapunov below)</text>',
    ]

    # bifurcation: for each r, plot the attractor points
    n_r = 520
    for i in range(n_r + 1):
        r = r_min + (r_max - r_min) * i / n_r
        pts = attractor(r, transient=800, samples=160)
        rx = RX(r)
        for x in pts:
            parts.append(f'<circle cx="{rx:.1f}" cy="{XY(x):.1f}" r="0.5" fill="#4dabf7" opacity="0.7"/>')

    # chaos-onset marker
    parts.append(f'<line x1="{RX(CHAOS_ONSET):.1f}" y1="{by1:.1f}" x2="{RX(CHAOS_ONSET):.1f}" y2="{by0:.1f}" '
                 f'stroke="#ff6b6b" stroke-width="0.8" stroke-dasharray="4 4" opacity="0.5"/>')
    parts.append(f'<text x="{RX(CHAOS_ONSET):.1f}" y="{by1-4:.1f}" fill="#ff6b6b" font-size="10" '
                 f'text-anchor="middle">chaos onset 3.57</text>')
    parts.append(f'<text x="{x0-6:.1f}" y="{XY(1.0)+3:.1f}" fill="#8b949e" font-size="9" text-anchor="end">1</text>')
    parts.append(f'<text x="{x0-6:.1f}" y="{XY(0.0)+3:.1f}" fill="#8b949e" font-size="9" text-anchor="end">0</text>')
    parts.append(f'<text x="{x0-40:.1f}" y="{(by0+by1)/2:.1f}" fill="#8b949e" font-size="10" '
                 f'transform="rotate(-90 {x0-40:.1f} {(by0+by1)/2:.1f})" text-anchor="middle">attractor x</text>')

    # bottom: Lyapunov exponent vs r
    ly0, ly1 = size - pad, split + 20
    lam_lo, lam_hi = -2.0, 1.0
    def LY(lam):
        lam = max(lam_lo, min(lam_hi, lam))
        return ly0 - (lam - lam_lo) / (lam_hi - lam_lo) * (ly0 - ly1)
    parts.append(f'<line x1="{x0}" y1="{LY(0.0):.1f}" x2="{x1}" y2="{LY(0.0):.1f}" '
                 f'stroke="#8b949e" stroke-width="0.8" stroke-dasharray="3 4" opacity="0.5"/>')
    lam_pts = []
    for i in range(n_r + 1):
        r = r_min + (r_max - r_min) * i / n_r
        lam_pts.append(f"{RX(r):.1f},{LY(lyapunov_exponent(r, transient=400, n=1500)):.1f}")
    parts.append(f'<polyline points="{" ".join(lam_pts)}" fill="none" stroke="#ff922b" stroke-width="1.4"/>')
    parts.append(f'<text x="{x0-6:.1f}" y="{LY(0.0)+3:.1f}" fill="#8b949e" font-size="9" text-anchor="end">0</text>')
    parts.append(f'<text x="{x0+90:.1f}" y="{LY(0.7):.1f}" fill="#ff922b" font-size="10">'
                 f'Lyapunov > 0 = chaos</text>')
    for r in (2.8, 3.2, 3.6, 4.0):
        parts.append(f'<text x="{RX(r):.1f}" y="{ly0+16:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="middle">{r:.1f}</text>')
    parts.append(f'<text x="{(x0+x1)/2:.1f}" y="{ly0+30:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">growth rate r</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
