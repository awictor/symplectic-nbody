"""Hessenberg demo: reduce a dense matrix to upper-Hessenberg form, shown as sparsity heatmaps (SVG)."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import hessenberg as HB


BG = "#0d1117"
TEXT = "#e6edf3"
GRAY = "#8b949e"
BLUE = "#4dabf7"
GREEN = "#06d6a0"


def _lcg(seed):
    st = seed & 0xFFFFFFFF

    def rnd():
        nonlocal st
        st = (1664525 * st + 1013904223) & 0xFFFFFFFF
        return (st >> 8) / (1 << 24)

    return rnd


def main(outdir=None):
    rnd = _lcg(20260913)
    n = 7
    A = [[rnd() * 2 - 1 for _ in range(n)] for _ in range(n)]
    H, Q = HB.hessenberg(A)

    lines = []
    lines.append("Hessenberg reduction: the eigensolver's first move")
    lines.append("=" * 52)
    lines.append(f"dense {n}x{n} matrix -> upper Hessenberg by Householder similarity (A = Q H Q^T)")
    lines.append("")
    lines.append("H (zeros below the subdiagonal):")
    for i in range(n):
        row = "  " + " ".join(f"{H[i][j]:+6.2f}" if abs(H[i][j]) > 1e-9 else "   .  " for j in range(n))
        lines.append(row)
    lines.append("")
    below = sum(1 for i in range(n) for j in range(n) if i > j + 1)
    lines.append(f"zeroed {below} of {n*n} entries (everything below the first subdiagonal)")
    lines.append("")
    check_h = HB.is_upper_hessenberg(H)
    R = HB.reconstruct(H, Q)
    err = max(abs(R[i][j] - A[i][j]) for i in range(n) for j in range(n))
    lines.append(f"upper Hessenberg?   {check_h}")
    lines.append(f"Q H Q^T == A?       max error {err:.2e}")
    lines.append(f"trace preserved?    A {HB.trace(A):+.5f}  H {HB.trace(H):+.5f}")
    lines.append("")
    # symmetric -> tridiagonal
    S = [[A[i][j] + A[j][i] for j in range(n)] for i in range(n)]
    Hs, Qs = HB.hessenberg(S)
    lines.append(f"a symmetric matrix reduces further, to TRIDIAGONAL: {HB.is_tridiagonal(Hs)}")
    lines.append("")
    lines.append("The QR algorithm preserves Hessenberg form, so each of its iterations then costs")
    lines.append("O(n^2) instead of O(n^3) -- this one-time reduction is why eigensolvers are fast.")

    text = "\n".join(lines)
    print(text)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        W, H_px = 720, 400
        cell = 34
        gap = 90

        def draw(mat, ox, title):
            mx = max(abs(mat[i][j]) for i in range(n) for j in range(n)) or 1
            oy = 60
            s_local = [f'<text x="{ox+n*cell/2:.0f}" y="{oy-10}" fill="{TEXT}" font-size="13" '
                       f'text-anchor="middle">{title}</text>']
            for i in range(n):
                for j in range(n):
                    v = mat[i][j]
                    if abs(v) < 1e-9:
                        col = "#161b22"
                    else:
                        inten = min(1.0, abs(v) / mx)
                        r = int(30 + inten * 47)
                        g = int(40 + inten * 130)
                        b = int(60 + inten * 190)
                        col = f"#{r:02x}{g:02x}{b:02x}"
                    s_local.append(f'<rect x="{ox+j*cell}" y="{oy+i*cell}" width="{cell-2}" '
                                   f'height="{cell-2}" fill="{col}"/>')
            return s_local

        s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H_px}" '
             f'viewBox="0 0 {W} {H_px}" font-family="monospace">']
        s.append(f'<rect width="{W}" height="{H_px}" fill="{BG}"/>')
        s.append(f'<text x="40" y="30" fill="{TEXT}" font-size="15">'
                 f'Householder similarity: dense A -> upper Hessenberg H</text>')
        s += draw(A, 40, "dense A")
        # arrow
        s.append(f'<text x="{40 + n*cell + 15}" y="{60 + n*cell/2}" fill="{GREEN}" '
                 f'font-size="24">-&gt;</text>')
        s += draw(H, 40 + n * cell + gap, "upper Hessenberg H")
        s.append(f'<text x="40" y="{H_px-14}" fill="{GRAY}" font-size="10">'
                 f'Every entry below the first subdiagonal is zeroed (dark), by reflectors applied on '
                 f'both sides so the eigenvalues are preserved.</text>')
        s.append("</svg>")
        with open(os.path.join(outdir, "hessenberg.svg"), "w", encoding="utf-8") as fh:
            fh.write("".join(s))

    return text


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
