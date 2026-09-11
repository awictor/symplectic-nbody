"""Demo: power iteration -- eigenvalues without the characteristic polynomial.

Finds the dominant eigenvalue by power iteration (watching the Rayleigh quotient converge),
targets an interior eigenvalue with shifted inverse iteration, and recovers a full symmetric
spectrum by deflation. Draws the Rayleigh-quotient convergence and the recovered spectrum.

    python examples/eigen_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eigen import (power_iteration, inverse_iteration, eigenvalues_symmetric,  # noqa: E402
                   rayleigh_quotient, residual_norm, trace, _matvec, _normalize)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    A = [[4, 1, 0, 0], [1, 3, 1, 0], [0, 1, 2, 1], [0, 0, 1, 1]]   # symmetric tridiagonal
    print("Power iteration: multiply by A and normalize -> dominant eigenvector\n")
    lam, v, it = power_iteration(A)
    print(f"  dominant eigenvalue = {lam:.6f}  (converged in {it} iterations)")
    print(f"  eigenvector = {[round(x, 4) for x in v]}")
    print(f"  residual ||A v - lambda v|| = {residual_norm(A, lam, v):.2e}\n")

    print("  Shifted inverse iteration targets any eigenvalue (the one nearest the shift):")
    for shift in (0.5, 2.0, 3.5):
        el, _, _ = inverse_iteration(A, shift)
        print(f"    nearest {shift}:  lambda = {el:.6f}")

    vals, _ = eigenvalues_symmetric(A)
    print(f"\n  Full spectrum by deflation: {[round(x, 4) for x in vals]}")
    print(f"  sum of eigenvalues = {sum(vals):.4f}  (trace = {trace(A)}) -- they match")
    print("\n  No characteristic polynomial, no root-finding: each multiply amplifies the")
    print("  largest-|lambda| direction, so the vector aligns with it. It is how PageRank ranks")
    print("  the web, PCA finds data axes, and vibration modes of a structure are computed.")

    _svg(os.path.join(outdir, "eigen.svg"), A, vals)
    print(f"\n  wrote {os.path.join(outdir, 'eigen.svg')}")


def _svg(path, A, vals, w=760, h=390):
    # left: Rayleigh-quotient convergence to the dominant eigenvalue
    n = len(A)
    v = _normalize([math.sin(1.0 + i) + 1.3 for i in range(n)])
    quotients = [rayleigh_quotient(A, v)]
    for _ in range(24):
        w_ = _matvec(A, v)
        nw = math.sqrt(sum(x * x for x in w_))
        v = [x / nw for x in w_]
        quotients.append(rayleigh_quotient(A, v))
    target = vals[0]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="monospace">',
        f'<rect width="{w}" height="{h}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Power iteration: the Rayleigh quotient climbs to the eigenvalue</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'estimate converging to the dominant eigenvalue (left); the full spectrum by deflation (right)</text>',
    ]

    lx0, lx1 = 55, w // 2 - 20
    ly0, ly1 = h - 55, 65
    qmin = min(quotients) - 0.1
    qmax = max(target, max(quotients)) + 0.1

    def LX(i):
        return lx0 + i / (len(quotients) - 1) * (lx1 - lx0)

    def LY(q):
        return ly0 - (q - qmin) / (qmax - qmin) * (ly0 - ly1)

    parts.append(f'<line x1="{lx0}" y1="{ly0}" x2="{lx1}" y2="{ly0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{lx0}" y1="{ly0}" x2="{lx0}" y2="{ly1}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{lx0}" y1="{LY(target):.1f}" x2="{lx1}" y2="{LY(target):.1f}" '
                 f'stroke="#06d6a0" stroke-width="1" stroke-dasharray="4 3"/>')
    parts.append(f'<text x="{lx1-2:.1f}" y="{LY(target)-4:.1f}" fill="#06d6a0" font-size="9" '
                 f'text-anchor="end">lambda = {target:.3f}</text>')
    pts = " ".join(f"{LX(i):.1f},{LY(q):.1f}" for i, q in enumerate(quotients))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#4dabf7" stroke-width="2"/>')
    for i, q in enumerate(quotients):
        parts.append(f'<circle cx="{LX(i):.1f}" cy="{LY(q):.1f}" r="1.8" fill="#4dabf7"/>')
    parts.append(f'<text x="{(lx0+lx1)/2:.1f}" y="{ly0+18:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">iteration -> Rayleigh quotient</text>')

    # right: the spectrum as vertical bars on a number line
    rx0, rx1 = w // 2 + 45, w - 30
    ry0, ry1 = h - 55, 90
    vmin = min(vals) - 0.5
    vmax = max(vals) + 0.5

    def RX(val):
        return rx0 + (val - vmin) / (vmax - vmin) * (rx1 - rx0)

    axis_y = (ry0 + ry1) / 2
    parts.append(f'<line x1="{rx0}" y1="{axis_y:.1f}" x2="{rx1}" y2="{axis_y:.1f}" stroke="#8b949e" stroke-width="1.2"/>')
    for val in vals:
        parts.append(f'<line x1="{RX(val):.1f}" y1="{axis_y-22:.1f}" x2="{RX(val):.1f}" y2="{axis_y+22:.1f}" '
                     f'stroke="#8338ec" stroke-width="2.5"/>')
        parts.append(f'<text x="{RX(val):.1f}" y="{axis_y-28:.1f}" fill="#b197fc" font-size="9" '
                     f'text-anchor="middle">{val:.2f}</text>')
    parts.append(f'<text x="{(rx0+rx1)/2:.1f}" y="{ry0+2:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">eigenvalues on the real line</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
