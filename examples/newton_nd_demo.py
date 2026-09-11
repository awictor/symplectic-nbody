"""Demo: Newton's method for nonlinear systems in n dimensions.

Solves a 2-D nonlinear system, showing Newton's quadratic convergence (correct digits roughly
doubling each step), how a damped line search rescues a start where plain Newton diverges, and how
Broyden's quasi-Newton trades per-step cost for a few more iterations. Draws the residual-norm
convergence curves.

    python examples/newton_nd_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from newton_nd import newton, newton_damped, broyden  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    # circle-line intersection: x^2 + y^2 = 4, y = x -> (sqrt2, sqrt2)
    def F(v):
        x, y = v
        return [x * x + y * y - 4, y - x]

    def J(v):
        x, y = v
        return [[2 * x, 2 * y], [-1, 1]]

    s2 = math.sqrt(2)
    print("Newton's method in n dimensions: J(x) delta = -F(x), x <- x + delta\n")
    print("  System: x^2 + y^2 = 4 and y = x  ->  root (sqrt2, sqrt2) = "
          f"({s2:.6f}, {s2:.6f})\n")

    root, it, conv, hist = newton(F, [1.6, 2.4], jacobian=J, track=True)
    print(f"  Newton from (1.6, 2.4): converged in {it} iterations to "
          f"({root[0]:.8f}, {root[1]:.8f})")
    print("  residual norm per step (note the digits roughly DOUBLING -- quadratic convergence):")
    for k, h in enumerate(hist):
        print(f"    step {k}: |F| = {h:.2e}")

    _, itb, _, histb = broyden(F, [1.6, 2.4], track=True)
    print(f"\n  Broyden quasi-Newton from the same start: {itb} iterations "
          f"(no Jacobian recompute per step; a bit slower to converge)")

    # a start where plain Newton diverges but damping rescues it
    def Fa(v):
        return [math.atan(v[0])]

    def Ja(v):
        return [[1.0 / (1.0 + v[0] ** 2)]]

    print("\n  Global robustness -- solving arctan(x) = 0 from a far start x0 = 5:")
    plain = "diverged (overshoots, |x| explodes)"
    try:
        r, i, c = newton(Fa, [5.0], jacobian=Ja, max_iter=30)
        if c:
            plain = f"converged to {r[0]:.6f}"
    except OverflowError:
        pass
    rd, idd, cd = newton_damped(Fa, [5.0], jacobian=Ja)
    print(f"    plain Newton:  {plain}")
    print(f"    damped Newton: converged to {rd[0]:.6f} in {idd} iterations "
          f"(backtracks until the residual drops)")

    # a 3-variable system for good measure
    def F3(v):
        x, y, z = v
        return [x + y + z - 6, x * x + y * y + z * z - 14, x * y * z - 6]

    r3, i3, c3 = newton(F3, [0.5, 1.5, 3.5])
    print(f"\n  3-variable system (sum 6, sq-sum 14, product 6): root "
          f"({r3[0]:.4f}, {r3[1]:.4f}, {r3[2]:.4f}) = (1, 2, 3)")

    print("\n  Each step linearizes the system at the current point and jumps to that linear model's")
    print("  root; near a solution the error squares each iteration. Far away it can overshoot, so a")
    print("  line search that backtracks until the residual falls buys global robustness. This is")
    print("  the solver inside circuit simulation, inverse kinematics, and chemical equilibrium.")

    _svg(os.path.join(outdir, "newton_nd.svg"), hist, histb)
    print(f"\n  wrote {os.path.join(outdir, 'newton_nd.svg')}")


def _svg(path, hist_newton, hist_broyden, width=760, height=400):
    lx0, lx1 = 60, width - 40
    y0, y1 = height - 55, 70
    all_h = [max(h, 1e-16) for h in hist_newton + hist_broyden]
    lo = math.log10(min(all_h))
    hi = math.log10(max(all_h))
    n = max(len(hist_newton), len(hist_broyden))

    def X(k):
        return lx0 + k / max(1, n - 1) * (lx1 - lx0)

    def Y(h):
        return y0 - (math.log10(max(h, 1e-16)) - lo) / (hi - lo) * (y0 - y1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Newton vs Broyden: residual norm per step (log scale)</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f"Newton's near-vertical drop is quadratic convergence (digits double); Broyden takes "
        f'a few more steps</text>',
        f'<line x1="{lx0}" y1="{y0}" x2="{lx1}" y2="{y0}" stroke="#8b949e" stroke-width="1.2"/>',
        f'<line x1="{lx0}" y1="{y0}" x2="{lx0}" y2="{y1}" stroke="#8b949e" stroke-width="1.2"/>',
    ]
    # y gridlines at powers of ten
    p = int(math.floor(lo))
    while p <= hi:
        yy = Y(10 ** p)
        parts.append(f'<line x1="{lx0}" y1="{yy:.1f}" x2="{lx1}" y2="{yy:.1f}" '
                     f'stroke="#21262d" stroke-width="1"/>')
        parts.append(f'<text x="{lx0-6:.1f}" y="{yy+4:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="end">1e{p}</text>')
        p += 3

    def curve(hist, color, label, ly):
        pts = " ".join(f"{X(k):.1f},{Y(hist[k]):.1f}" for k in range(len(hist)))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.2"/>')
        for k in range(len(hist)):
            parts.append(f'<circle cx="{X(k):.1f}" cy="{Y(hist[k]):.1f}" r="3" fill="{color}"/>')
        parts.append(f'<text x="{lx1-4:.1f}" y="{ly:.1f}" fill="{color}" font-size="11" '
                     f'text-anchor="end">{label}</text>')

    curve(hist_broyden, "#ff922b", "Broyden", y1 + 14)
    curve(hist_newton, "#4dabf7", "Newton (quadratic)", y1)
    parts.append(f'<text x="{(lx0+lx1)/2:.1f}" y="{y0+34:.1f}" fill="#8b949e" font-size="11" '
                 f'text-anchor="middle">iteration</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
