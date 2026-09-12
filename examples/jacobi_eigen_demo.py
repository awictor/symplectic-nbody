"""Demo: the Jacobi eigenvalue algorithm -- diagonalising a symmetric matrix by rotations.

Diagonalises symmetric matrices, verifies reconstruction and the trace/determinant identities, and
shows the off-diagonal norm shrinking toward zero rotation by rotation. Draws that convergence.

    python examples/jacobi_eigen_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from jacobi_eigen import (sorted_eigen, reconstruct, is_orthogonal, matvec,  # noqa: E402
                          _off_diagonal_norm_sq)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Jacobi eigenvalue algorithm: A = V D V^T for symmetric A, via rotations\n")

    A = [[4, 1, -2, 2], [1, 2, 0, 1], [-2, 0, 3, -2], [2, 1, -2, -1]]
    vals, V = sorted_eigen(A)
    print("  symmetric matrix A:")
    for row in A:
        print("    " + "  ".join(f"{v:5.1f}" for v in row))
    print(f"\n  eigenvalues (descending): {[round(v, 4) for v in vals]}")
    print(f"  eigenvectors orthonormal: {is_orthogonal(V)}")
    recon = reconstruct(vals, V)
    err = max(abs(recon[i][j] - A[i][j]) for i in range(4) for j in range(4))
    print(f"  reconstruction error |V D V^T - A|: {err:.2e}")
    print(f"  sum of eigenvalues = {sum(vals):.4f}  (trace = {sum(A[i][i] for i in range(4))})")

    print("\n  each eigenpair satisfies A v = lambda v:")
    for c in range(4):
        v = [V[r][c] for r in range(4)]
        Av = matvec(A, v)
        residual = max(abs(Av[i] - vals[c] * v[i]) for i in range(4))
        print(f"    lambda = {vals[c]:8.4f}   |A v - lambda v| = {residual:.1e}")

    # trace the off-diagonal norm shrinking
    trace = _trace_offdiag(A)
    print(f"\n  off-diagonal Frobenius norm falls from {math.sqrt(trace[0]):.3f} to "
          f"{math.sqrt(trace[-1]):.1e} over {len(trace)-1} rotations")
    print("    each rotation zeros the largest off-diagonal entry; the total off-diagonal mass")
    print("    strictly decreases every step, driving A to diagonal form. Slower than QR for big")
    print("    matrices, but simple, robust, and accurate even for the tiniest eigenvalues.")

    _svg(os.path.join(outdir, "jacobi_eigen.svg"), trace)
    print(f"\n  wrote {os.path.join(outdir, 'jacobi_eigen.svg')}")


def _trace_offdiag(A):
    """Run Jacobi manually, recording the off-diagonal norm-squared after each rotation."""
    n = len(A)
    a = [[float(A[i][j]) for j in range(n)] for i in range(n)]
    trace = [_off_diagonal_norm_sq(a)]
    for _ in range(200):
        p, q, largest = 0, 1, 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(a[i][j]) > largest:
                    largest = abs(a[i][j]); p, q = i, j
        if largest < 1e-14:
            break
        app, aqq, apq = a[p][p], a[q][q], a[p][q]
        phi = (aqq - app) / (2.0 * apq)
        t = (1.0 if phi >= 0 else -1.0) / (abs(phi) + math.sqrt(phi * phi + 1.0))
        c = 1.0 / math.sqrt(t * t + 1.0)
        s = t * c
        for k in range(n):
            akp, akq = a[k][p], a[k][q]
            a[k][p] = c * akp - s * akq
            a[k][q] = s * akp + c * akq
        for k in range(n):
            apk, aqk = a[p][k], a[q][k]
            a[p][k] = c * apk - s * aqk
            a[q][k] = s * apk + c * aqk
        trace.append(_off_diagonal_norm_sq(a))
    return trace


def _svg(path, trace, width=740, height=360):
    n = len(trace)
    ox, oy = 60, 300
    pw, ph = width - 100, 240
    import math as m
    # plot log10 of off-diagonal norm (sqrt of norm-sq)
    vals = [max(1e-18, t) ** 0.5 for t in trace]
    logs = [m.log10(v) for v in vals]
    lmin, lmax = min(logs), max(logs)

    def px(i):
        return ox + i / max(1, n - 1) * pw

    def py(lv):
        return oy - (lv - lmin) / (lmax - lmin + 1e-12) * ph

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="17">'
        f'Jacobi convergence: off-diagonal norm vs rotation (log scale)</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'each rotation zeros the largest off-diagonal entry; the norm plunges toward 0 (diagonal)'
        f'</text>',
    ]
    parts.append(f'<line x1="{ox}" y1="{oy}" x2="{ox+pw}" y2="{oy}" stroke="#30363d"/>')
    parts.append(f'<line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy-ph}" stroke="#30363d"/>')
    pts = " ".join(f"{px(i):.1f},{py(lv):.1f}" for i, lv in enumerate(logs))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#06d6a0" stroke-width="2"/>')
    for i, lv in enumerate(logs):
        parts.append(f'<circle cx="{px(i):.1f}" cy="{py(lv):.1f}" r="3" fill="#4dabf7"/>')
    parts.append(f'<text x="{ox+pw/2:.0f}" y="{oy+30}" fill="#8b949e" font-size="12" '
                 f'text-anchor="middle">rotation number -></text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
