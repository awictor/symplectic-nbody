"""Demo: the linear-quadratic regulator -- optimal feedback that stabilizes a system.

Designs an LQR controller for a double-integrator (a cart you push, wanting position + velocity to
zero) and for a naturally unstable system, shows the closed-loop gains and stability, and draws the
state trajectories decaying to the origin under optimal control -- plus how the control-effort penalty
trades speed for gentleness.

    python examples/lqr_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lqr import (lqr_gain, closed_loop_matrix, simulate, lqr_cost,  # noqa: E402
                 spectral_radius_2x2)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("LQR: the optimal feedback gain from the Riccati equation\n")

    dt = 0.1
    A = [[1.0, dt], [0.0, 1.0]]          # double integrator: position, velocity
    B = [[0.5 * dt * dt], [dt]]
    Q = [[1.0, 0.0], [0.0, 1.0]]

    print("  double integrator (push a cart to bring position & velocity to zero):")
    for r in (0.01, 0.1, 1.0, 10.0):
        K, P = lqr_gain(A, B, Q, [[r]])
        rho = spectral_radius_2x2(closed_loop_matrix(A, B, K))
        cost = lqr_cost(A, B, K, Q, [[r]], [5.0, 0.0], 300)
        print(f"    R={r:5.2f}: gain K=[{K[0][0]:6.3f}, {K[0][1]:6.3f}]  "
              f"closed-loop rho={rho:.3f}  cost={cost:8.2f}")
    print("    (small R = cheap control = aggressive gain, fast settle; large R = gentle)\n")

    # unstable system
    Au = [[1.5, 1.0], [0.0, 1.2]]
    Bu = [[0.0], [1.0]]
    Ku, _ = lqr_gain(Au, Bu, [[1.0, 0], [0, 1.0]], [[1.0]])
    print("  naturally unstable system (both open-loop modes > 1):")
    print(f"    open-loop spectral radius:  {spectral_radius_2x2(Au):.3f}  (unstable)")
    print(f"    closed-loop spectral radius: {spectral_radius_2x2(closed_loop_matrix(Au, Bu, Ku)):.3f}"
          f"  (stable under LQR)\n")

    print("  The optimal control u = -K x uses a constant gain K built from the Riccati solution P.")
    print("  It provably minimises the total quadratic cost -- state error plus control effort -- over")
    print("  all time, which is why LQR is the inner loop of drones, rockets, and robots everywhere.")

    _svg(os.path.join(outdir, "lqr.svg"), A, B, Q)
    print(f"\n  wrote {os.path.join(outdir, 'lqr.svg')}")


def _svg(path, A, B, Q, width=760, height=400):
    # position trajectories from x0=[5,0] for several R values
    x0 = [5.0, 0.0]
    steps = 120
    curves = []
    for r, col in ((0.01, "#06d6a0"), (0.1, "#4dabf7"), (1.0, "#ffd43b"), (10.0, "#ff922b")):
        K, _ = lqr_gain(A, B, Q, [[r]])
        xs = simulate(A, B, K, x0, steps)
        curves.append((r, col, [x[0] for x in xs]))    # position over time

    ox, oy = 55, 330
    pw, ph = width - 90, 260
    ymax = 5.5
    ymin = -2.5
    span = ymax - ymin

    def px(i):
        return ox + i / steps * pw

    def py(v):
        return oy - (v - ymin) / span * ph

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="17">'
        f'LQR: position settling to zero under different control penalties R</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'cheaper control (small R) settles faster; expensive control (large R) is gentler</text>',
    ]
    # zero line + start
    parts.append(f'<line x1="{ox}" y1="{py(0):.1f}" x2="{ox+pw}" y2="{py(0):.1f}" '
                 f'stroke="#30363d" stroke-width="1"/>')
    parts.append(f'<text x="{ox-8}" y="{py(0)+4:.0f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="end">0</text>')
    parts.append(f'<text x="{ox-8}" y="{py(5)+4:.0f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="end">5</text>')
    for k, (r, col, pos) in enumerate(curves):
        pts = " ".join(f"{px(i):.1f},{py(pos[i]):.1f}" for i in range(len(pos)))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2"/>')
        parts.append(f'<text x="{ox+pw-70}" y="{60+18*k}" fill="{col}" font-size="11">R={r}</text>')
    parts.append(f'<text x="{ox+pw/2:.0f}" y="{oy+28}" fill="#8b949e" font-size="12" '
                 f'text-anchor="middle">time step</text>')
    parts.append(f'<text x="20" y="{oy-ph-2:.0f}" fill="#8b949e" font-size="11">cart position</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
