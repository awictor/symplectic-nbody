"""Demo: adaptive Dormand-Prince RK45 solving ODEs with automatic step control.

Solves exponential decay, a harmonic oscillator, and the Van der Pol oscillator (stiff-ish, with
sharp transitions), showing the adaptive step size shrinking where the solution changes fastest.
Draws the Van der Pol solution with step sizes marked.

    python examples/rk45_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rk45 import solve  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Dormand-Prince RK45: adaptive-step ODE solving\n")

    # exponential decay
    ts, ys = solve(lambda t, y: -y, 0.0, 1.0, 5.0, atol=1e-9, rtol=1e-9)
    print(f"  y' = -y, y(0)=1:  y(5) = {ys[-1]:.8f}  (exact e^-5 = {math.exp(-5):.8f})")
    print(f"    error {abs(ys[-1]-math.exp(-5)):.2e} in {len(ts)-1} adaptive steps")

    # harmonic oscillator energy conservation
    ts, ys = solve(lambda t, y: [y[1], -y[0]], 0.0, [1.0, 0.0], 20 * math.pi,
                   atol=1e-10, rtol=1e-10)
    energies = [0.5 * (s[0] ** 2 + s[1] ** 2) for s in ys]
    print(f"\n  Harmonic oscillator over 10 periods:")
    print(f"    energy drift {max(abs(e-0.5) for e in energies):.2e} (should be ~0), {len(ts)-1} steps")

    # tolerance vs steps vs error
    print("\n  Tolerance controls accuracy AND effort:")
    for tol in [1e-4, 1e-6, 1e-8, 1e-10]:
        ts, ys = solve(lambda t, y: -y, 0.0, 1.0, 5.0, atol=tol, rtol=tol)
        print(f"    tol={tol:.0e}: {len(ts)-1:3d} steps, error {abs(ys[-1]-math.exp(-5)):.2e}")

    # Van der Pol oscillator: nonlinear, sharp transitions -> adaptive stepping shines
    mu = 5.0
    def vdp(t, y):
        return [y[1], mu * (1 - y[0] ** 2) * y[1] - y[0]]
    ts, ys = solve(vdp, 0.0, [2.0, 0.0], 25.0, atol=1e-7, rtol=1e-7)
    steps = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
    print(f"\n  Van der Pol oscillator (mu={mu}, nonlinear relaxation oscillation):")
    print(f"    {len(ts)-1} adaptive steps; step size ranges "
          f"{min(steps):.4f} to {max(steps):.4f}")
    print(f"    -> the solver takes {max(steps)/min(steps):.0f}x bigger steps in the smooth stretches")
    print(f"       than through the sharp switch-backs -- exactly where a fixed step would waste work.")

    print("\n  Each step computes a 5th- and a 4th-order estimate from shared stage evaluations; their")
    print("  difference estimates the local error, which accepts/rejects the step and rescales it by")
    print("  (tol/error)^(1/5). Big steps in calm regions, tiny steps through rapid change.")

    _svg(os.path.join(outdir, "rk45.svg"), ts, ys)
    print(f"\n  wrote {os.path.join(outdir, 'rk45.svg')}")


def _svg(path, ts, ys, width=760, height=440):
    m_left, m_bot, m_top, m_right = 60, 90, 80, 40
    pw = width - m_left - m_right
    ph = height - m_top - m_bot

    xvals = [s[0] for s in ys]
    tmax = ts[-1]
    ymin, ymax = min(xvals), max(xvals)
    yr = ymax - ymin or 1

    def px(t):
        return m_left + t / tmax * pw

    def py(v):
        return m_top + ph - (v - ymin) / yr * ph

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">'
        f'Adaptive RK45: Van der Pol oscillator, with step markers</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'blue = solution x(t); ticks below = accepted steps (dense where the solution swings fast)</text>',
    ]

    parts.append(f'<line x1="{m_left}" y1="{py(0):.1f}" x2="{m_left+pw}" y2="{py(0):.1f}" '
                 f'stroke="#30363d" stroke-width="1"/>')

    pts = " ".join(f"{px(ts[i]):.1f},{py(ys[i][0]):.1f}" for i in range(len(ts)))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#4dabf7" stroke-width="1.8"/>')

    # step-size ticks along the bottom
    tick_y = m_top + ph + 20
    for t in ts:
        parts.append(f'<line x1="{px(t):.1f}" y1="{tick_y}" x2="{px(t):.1f}" y2="{tick_y+10}" '
                     f'stroke="#ff922b" stroke-width="0.8"/>')
    parts.append(f'<text x="{m_left}" y="{tick_y+26}" fill="#8b949e" font-size="10">'
                 f'accepted steps ({len(ts)-1} total) -- clustered at the sharp transitions</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
