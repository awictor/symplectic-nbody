"""Schur demo: reduce a matrix to real Schur form, read eigenvalues off the diagonal blocks (SVG)."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import schur_decomposition as SD


BG = "#0d1117"
TEXT = "#e6edf3"
GRAY = "#8b949e"
GREEN = "#06d6a0"
YELLOW = "#ffd43b"
RED = "#ff6b6b"


def _lcg(seed):
    st = seed & 0xFFFFFFFF

    def rnd():
        nonlocal st
        st = (1664525 * st + 1013904223) & 0xFFFFFFFF
        return (st >> 8) / (1 << 24)

    return rnd


def main(outdir=None):
    # a matrix with both real and complex eigenvalues
    rnd = _lcg(20260913)
    n = 6
    A = [[rnd() * 2 - 1 for _ in range(n)] for _ in range(n)]
    Q, T = SD.schur(A)
    eigs = SD.eigenvalues_from_schur(T)

    lines = []
    lines.append("Real Schur decomposition: A = Q T Q^T")
    lines.append("=" * 52)
    lines.append(f"{n}x{n} matrix reduced to real Schur form by shifted QR iteration")
    lines.append("")
    lines.append("T (quasi-upper-triangular; 2x2 blocks carry complex pairs):")
    for i in range(n):
        row = "  " + " ".join(f"{T[i][j]:+6.2f}" if abs(T[i][j]) > 1e-9 else "   .  " for j in range(n))
        lines.append(row)
    lines.append("")
    lines.append("eigenvalues read from the diagonal blocks:")
    for z in sorted(eigs, key=lambda z: (round(z.real, 4), round(z.imag, 4))):
        kind = "real" if abs(z.imag) < 1e-9 else "complex pair"
        lines.append(f"  {z.real:+.4f}{z.imag:+.4f}i   ({kind})")
    lines.append("")
    R = SD.reconstruct(Q, T)
    err = max(abs(R[i][j] - A[i][j]) for i in range(n) for j in range(n))
    check_orth = all(abs(sum(Q[k][i] * Q[k][j] for k in range(n)) - (1.0 if i == j else 0.0)) < 1e-8
                     for i in range(n) for j in range(n))
    lines.append(f"Q orthogonal?      {check_orth}")
    lines.append(f"Q T Q^T == A?      max error {err:.2e}")
    lines.append("")
    lines.append("The Schur form always exists over the reals and uses only stable orthogonal")
    lines.append("transforms -- no ill-conditioned eigenvector basis. It is how eigenvalues are found.")

    text = "\n".join(lines)
    print(text)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        W, H = 720, 400
        cell = 40
        gap = 80

        def draw(mat, ox, title, blocks=None):
            oy = 60
            mx = max(abs(mat[i][j]) for i in range(n) for j in range(n)) or 1
            out = [f'<text x="{ox+n*cell/2:.0f}" y="{oy-10}" fill="{TEXT}" font-size="13" '
                   f'text-anchor="middle">{title}</text>']
            for i in range(n):
                for j in range(n):
                    v = mat[i][j]
                    if abs(v) < 1e-9:
                        col = "#161b22"
                    else:
                        inten = min(1.0, abs(v) / mx)
                        col = f"#{int(30+inten*47):02x}{int(40+inten*130):02x}{int(60+inten*190):02x}"
                    out.append(f'<rect x="{ox+j*cell}" y="{oy+i*cell}" width="{cell-2}" '
                               f'height="{cell-2}" fill="{col}"/>')
            return out

        s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
        s.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
        s.append(f'<text x="40" y="30" fill="{TEXT}" font-size="15">'
                 f'Shifted QR iteration: dense A -> real Schur form T</text>')
        s += draw(A, 40, "dense A")
        s.append(f'<text x="{40 + n*cell + 12}" y="{60 + n*cell/2}" fill="{GREEN}" '
                 f'font-size="24">-&gt;</text>')
        oxT = 40 + n * cell + gap
        s += draw(T, oxT, "real Schur form T")
        # outline the 2x2 blocks in T (consecutive nonzero subdiagonals) in red
        oy = 60
        i = 0
        while i < n:
            if i + 1 < n and abs(T[i + 1][i]) > 1e-6 * (abs(T[i][i]) + abs(T[i + 1][i + 1])):
                s.append(f'<rect x="{oxT+i*cell:.0f}" y="{oy+i*cell:.0f}" width="{2*cell-2}" '
                         f'height="{2*cell-2}" fill="none" stroke="{RED}" stroke-width="2"/>')
                i += 2
            else:
                i += 1
        s.append(f'<text x="40" y="{H-14}" fill="{GRAY}" font-size="10">'
                 f'T is upper triangular except for 2x2 blocks (red) that hold complex-conjugate '
                 f'eigenvalue pairs; 1x1 diagonal entries are real eigenvalues.</text>')
        s.append("</svg>")
        with open(os.path.join(outdir, "schur_decomposition.svg"), "w", encoding="utf-8") as fh:
            fh.write("".join(s))

    return text


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
