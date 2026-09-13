"""Demo: the QR algorithm driving a matrix to triangular form, eigenvalues emerging on the diagonal.

Reduces a matrix to Hessenberg form, runs the shifted QR iteration, and shows the subdiagonal entries
decaying to zero as the eigenvalues settle onto the diagonal -- then reports the eigenvalues (real and
complex) and cross-checks them against the characteristic-polynomial roots. Draws the subdiagonal
magnitudes shrinking over iterations.

    python examples/qr_algorithm_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qr_algorithm import (  # noqa: E402
    hessenberg,
    eigenvalues,
    characteristic_poly_roots,
    trace,
    _qr_decompose_square,
    _matmul,
)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("The QR algorithm: factor A = QR, reform RQ, repeat -> eigenvalues on the diagonal\n")

    # a symmetric matrix with well-separated eigenvalue magnitudes, so plain (unshifted) QR
    # converges cleanly and the subdiagonal decay is visible
    A = [
        [10.0, 2.0, 1.0, 0.0],
        [2.0, 6.0, 1.0, 0.5],
        [1.0, 1.0, 3.0, 0.3],
        [0.0, 0.5, 0.3, 1.0],
    ]
    n = len(A)
    print(f"  {n}x{n} symmetric matrix, trace = {trace(A):.1f}")

    # instrument the unshifted iteration to show the subdiagonal decaying
    H = hessenberg(A)
    history = []
    M = [row[:] for row in H]
    for it in range(40):
        subdiag = [abs(M[i + 1][i]) for i in range(n - 1)]
        history.append(max(subdiag))
        Q, R = _qr_decompose_square(M)
        M = _matmul(R, Q)

    print(f"\n  max |subdiagonal| entry as the iteration proceeds (convergence to triangular):")
    for it in [0, 2, 5, 10, 20, 39]:
        bar = "#" * int(50 * history[it] / history[0]) if history[0] > 0 else ""
        print(f"    iter {it:>3}: {history[it]:.2e}  {bar}")

    eigs = eigenvalues(A)
    eigs_sorted = sorted(eigs, key=lambda z: (z.real if isinstance(z, complex) else z))
    print(f"\n  eigenvalues (shifted QR with deflation):")
    for e in eigs_sorted:
        if isinstance(e, complex) and abs(e.imag) > 1e-9:
            print(f"    {e.real:+.5f} {'+' if e.imag >= 0 else '-'} {abs(e.imag):.5f}i")
        else:
            print(f"    {(e.real if isinstance(e, complex) else e):+.5f}")
    s = sum(e for e in eigs)
    print(f"\n  sum of eigenvalues = {complex(s).real:.5f}  (matches trace {trace(A):.1f})")

    # a nonsymmetric matrix with a complex pair
    B = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 2.0]]
    be = eigenvalues(B)
    cp = characteristic_poly_roots(B)
    print(f"\n  nonsymmetric example (rotation + scale): eigenvalues {[_fmt(z) for z in be]}")
    print(f"  characteristic-poly roots cross-check:      {[_fmt(z) for z in cp]}")

    _svg(os.path.join(outdir, "qr_algorithm.svg"), history)
    print(f"\n  wrote {os.path.join(outdir, 'qr_algorithm.svg')}")


def _fmt(z):
    if isinstance(z, complex) and abs(z.imag) > 1e-9:
        return f"{z.real:.2f}{'+' if z.imag >= 0 else '-'}{abs(z.imag):.2f}i"
    return f"{(z.real if isinstance(z, complex) else z):.2f}"


def _svg(path, history, width=760, height=380):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="15">'
        f'QR iteration: max subdiagonal magnitude -&gt; 0 (matrix converging to triangular)</text>',
    ]
    ox, oy, ow, oh = 60, 55, width - 100, height - 110
    nit = len(history)
    # log scale for the y axis
    floor = 1e-16
    logs = [math.log10(max(h, floor)) for h in history]
    lo = min(logs)
    hi = max(logs)

    def px(i):
        return ox + ow * i / (nit - 1)

    def py(lg):
        return oy + oh * (1 - (lg - lo) / (hi - lo + 1e-12))

    parts.append(f'<rect x="{ox}" y="{oy}" width="{ow}" height="{oh}" fill="none" stroke="#30363d"/>')
    # decade gridlines
    d = math.floor(lo)
    while d <= hi:
        y = py(d)
        parts.append(f'<line x1="{ox}" y1="{y:.1f}" x2="{ox+ow}" y2="{y:.1f}" stroke="#161b22"/>')
        parts.append(f'<text x="{ox-6}" y="{y+3:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="end">1e{int(d)}</text>')
        d += 1
    pts = " ".join(f"{px(i):.1f},{py(logs[i]):.1f}" for i in range(nit))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#4dabf7" stroke-width="2"/>')
    for i in range(nit):
        parts.append(f'<circle cx="{px(i):.1f}" cy="{py(logs[i]):.1f}" r="2" fill="#ffd43b"/>')
    parts.append(f'<text x="{ox+ow/2:.0f}" y="{oy+oh+24:.0f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">QR iteration (log-scale subdiagonal magnitude)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
