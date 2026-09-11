"""Demo: the Lorenz attractor -- the butterfly effect made visible.

Prints the fixed points, volume-contraction rate and Lyapunov exponent, then draws the
butterfly attractor (x-z projection) and the divergence of two trajectories started a
millionth apart -- sensitive dependence on initial conditions.

    python examples/lorenz_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lorenz import (trajectory, volume_contraction_rate, fixed_points,  # noqa: E402
                    largest_lyapunov, rk4_step, SIGMA, BETA, RHO)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Lorenz system (sigma=%.0f, beta=8/3, rho=%.0f): toy weather that never repeats\n"
          % (SIGMA, RHO))
    print("  Fixed points:")
    for fp in fixed_points():
        print(f"    ({fp[0]:+.3f}, {fp[1]:+.3f}, {fp[2]:+.3f})")
    print("\n  Phase-space volume contraction rate div F = %.3f (dissipative)"
          % volume_contraction_rate())
    lam = largest_lyapunov(n=12000)
    print("  Largest Lyapunov exponent = %.3f (positive -> chaos)." % lam)
    print("  Prediction error grows e^(%.2f t): 10x every ~%.1f time units.\n"
          % (lam, math.log(10) / lam))

    # Show two nearby trajectories diverging.
    a = (1.0, 1.0, 1.0)
    b = (1.0 + 1e-6, 1.0, 1.0)
    print("  Two starts a millionth apart, separation over time:")
    for step_i in range(0, 4001, 800):
        if step_i > 0:
            for _ in range(800):
                a = rk4_step(a, 0.01)
                b = rk4_step(b, 0.01)
        sep = math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))
        print(f"    t = {step_i*0.01:>5.1f}  ->  separation {sep:.3e}")

    print("\n  A millionth of a degree in the initial state grows to opposite wings of the")
    print("  butterfly -- why weather is unforecastable beyond ~two weeks. Deterministic, yet")
    print("  unpredictable: the discovery that founded chaos theory.")

    _svg(os.path.join(outdir, "lorenz.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'lorenz.svg')}")


def _svg(path, size=760, pad=60):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="30" fill="#e6edf3" font-size="18">The Lorenz attractor</text>',
        f'<text x="20" y="48" fill="#8b949e" font-size="12">'
        f'the butterfly (x-z projection, top); two trajectories a millionth apart diverging (bottom)</text>',
    ]

    x0, x1 = pad, size - pad
    split = size * 0.60

    # --- top: butterfly x-z projection ---
    ty0, ty1 = split - 20, pad + 30
    traj = trajectory((1.0, 1.0, 1.0), 0.008, 12000)[1000:]
    xmin, xmax = -22, 22
    zmin, zmax = 0, 50
    def PX(x):
        return x0 + (x - xmin) / (xmax - xmin) * (x1 - x0)
    def PZ(z):
        return ty0 - (z - zmin) / (zmax - zmin) * (ty0 - ty1)
    poly = " ".join(f"{PX(p[0]):.1f},{PZ(p[2]):.1f}" for p in traj[::2])
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#4dabf7" stroke-width="0.4" opacity="0.7"/>')
    # mark the two convection fixed points
    for fp in fixed_points()[1:]:
        parts.append(f'<circle cx="{PX(fp[0]):.1f}" cy="{PZ(fp[2]):.1f}" r="3" fill="#ffd43b"/>')
    parts.append(f'<text x="{(x0+x1)/2:.1f}" y="{ty0+16:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">x-z projection (yellow = convection fixed points C+/-)</text>')

    # --- bottom: separation of two nearby trajectories (log scale) ---
    by0, by1 = size - pad, split + 20
    a = (1.0, 1.0, 1.0)
    b = (1.0 + 1e-6, 1.0, 1.0)
    dt, N = 0.01, 3500
    seps = []
    for _ in range(N):
        a = rk4_step(a, dt)
        b = rk4_step(b, dt)
        s = math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))
        seps.append(max(s, 1e-12))
    t_max = N * dt
    lo, hi = math.log10(min(seps)), math.log10(max(seps) * 1.2)
    def SX(t):
        return x0 + t / t_max * (x1 - x0)
    def SY(s):
        return by0 - (math.log10(s) - lo) / (hi - lo) * (by0 - by1)
    parts.append(f'<line x1="{x0}" y1="{by0}" x2="{x1}" y2="{by0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{x0}" y1="{by0}" x2="{x0}" y2="{by1}" stroke="#8b949e" stroke-width="1.2"/>')
    spts = " ".join(f"{SX(i*dt):.1f},{SY(seps[i]):.1f}" for i in range(0, N, 4))
    parts.append(f'<polyline points="{spts}" fill="none" stroke="#ff6b6b" stroke-width="1.8"/>')
    for e in range(int(math.floor(lo)), int(math.ceil(hi)) + 1, 2):
        gy = SY(10.0 ** e)
        if by1 <= gy <= by0:
            parts.append(f'<text x="{x0-6:.1f}" y="{gy+3:.1f}" fill="#8b949e" font-size="9" '
                         f'text-anchor="end">10^{e}</text>')
    parts.append(f'<text x="{(x0+x1)/2:.1f}" y="{by0+16:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">time (separation of two starts 1e-6 apart, log scale)</text>')
    parts.append(f'<text x="{x0+80:.1f}" y="{by1+4:.1f}" fill="#ff6b6b" font-size="10">'
                 f'exponential growth = the butterfly effect</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
