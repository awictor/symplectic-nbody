"""Demo: Thomas algorithm + Crank-Nicolson -- solving the heat equation in O(n) per step.

Diffuses a hot spike on a bar with Crank-Nicolson (unconditionally stable), compares the spreading
profile to the analytic sqrt(t)-broadening Gaussian, and shows the explicit scheme blowing up at the
same large time step. Draws the diffusing profile at several times.

    python examples/thomas_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from thomas import diffuse, total_heat, stability_ratio, heat_step_explicit  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Thomas + Crank-Nicolson: implicit heat diffusion, O(n) per step, any time step\n")

    L = 10.0
    N = 201
    dx = 2 * L / (N - 1)
    xs = [-L + i * dx for i in range(N)]
    alpha = 1.0
    sigma0 = 0.8
    u0 = [math.exp(-x * x / (2 * sigma0 ** 2)) for x in xs]
    dt = 0.1
    r = stability_ratio(alpha, dt, dx)
    print(f"  Bar of {N} points, alpha={alpha}, dt={dt}, dx={dx:.3f}")
    print(f"  Stability ratio r = alpha dt/dx^2 = {r:.2f}  (explicit needs r <= 0.5)\n")

    print(f"  {'time':>6}  {'peak (CN)':>10}  {'peak (analytic)':>16}  {'total heat':>11}")
    area0 = total_heat(u0, dx)
    snapshots = []
    u = list(u0)
    for step_group in range(5):
        t = step_group * 10 * dt
        var_t = sigma0 ** 2 + 2 * alpha * t
        peak_a = area0 / math.sqrt(2 * math.pi * var_t)
        print(f"  {t:>6.1f}  {max(u):>10.4f}  {peak_a:>16.4f}  {total_heat(u, dx):>11.4f}")
        snapshots.append((t, list(u)))
        u = diffuse(u, alpha, dt, dx, 10, scheme="crank_nicolson")

    # explicit blow-up at the same dt
    ue = diffuse(u0, alpha, dt, dx, 30, scheme="explicit")
    print(f"\n  Explicit scheme at the same dt after 30 steps: max |u| = {max(abs(v) for v in ue):.1e}")
    print("  -> the explicit scheme oscillates and diverges (r > 0.5); Crank-Nicolson stays smooth.")

    print("\n  Each Crank-Nicolson step solves a tridiagonal system by the Thomas algorithm in O(n),")
    print("  so the whole simulation is linear-time per step no matter how large the step.")

    _svg(os.path.join(outdir, "thomas.svg"), xs, snapshots)
    print(f"\n  wrote {os.path.join(outdir, 'thomas.svg')}")


def _svg(path, xs, snapshots, width=760, height=410):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        '<text x="20" y="26" fill="#e6edf3" font-size="15">'
        'Heat spike diffusing (Crank-Nicolson): profile spreads and flattens over time</text>',
    ]
    ox, oy, ow, oh = 50, 50, width - 90, height - 100
    xmin, xmax = xs[0], xs[-1]
    ymax = max(max(u) for _, u in snapshots) * 1.1

    def px(x):
        return ox + ow * (x - xmin) / (xmax - xmin)

    def py(y):
        return oy + oh * (1 - y / ymax)

    cols = ["#ff6b6b", "#ff922b", "#ffd43b", "#06d6a0", "#4dabf7"]
    parts.append(f'<line x1="{ox}" y1="{oy+oh}" x2="{ox+ow}" y2="{oy+oh}" stroke="#8b949e"/>')
    for idx, (t, u) in enumerate(snapshots):
        col = cols[idx % len(cols)]
        p = " ".join(f"{px(xs[i]):.1f},{py(u[i]):.1f}" for i in range(len(xs)))
        parts.append(f'<polyline points="{p}" fill="none" stroke="{col}" stroke-width="1.8"/>')
        parts.append(f'<text x="{ox+ow-90}" y="{oy+14+idx*15}" fill="{col}" font-size="10">'
                     f't = {t:.1f}</text>')
    parts.append(f'<text x="{ox+ow/2:.0f}" y="{oy+oh+20}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">position x</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
