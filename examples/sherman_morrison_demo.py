"""Sherman-Morrison-Woodbury demo: rank-1 inverse updates vs full re-inversion, accuracy + op-count (SVG)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import sherman_morrison as SM
from lu import inverse, determinant
from linsolve import solve


BG = "#0d1117"
TEXT = "#e6edf3"
GRAY = "#8b949e"
BLUE = "#4dabf7"
GREEN = "#06d6a0"
YELLOW = "#ffd43b"


def _lcg(seed):
    st = seed & 0xFFFFFFFF

    def rnd():
        nonlocal st
        st = (1664525 * st + 1013904223) & 0xFFFFFFFF
        return (st >> 8) / (1 << 24)

    return rnd


def main(outdir=None):
    rnd = _lcg(20260913)
    n = 8
    A = [[rnd() * 2 - 1 + (5 if i == j else 0) for j in range(n)] for i in range(n)]
    A_inv = inverse(A)
    u = [rnd() for _ in range(n)]
    v = [rnd() for _ in range(n)]
    mod = [[A[i][j] + u[i] * v[j] for j in range(n)] for i in range(n)]

    sm = SM.sherman_morrison_inverse(A_inv, u, v)
    full = inverse(mod)
    err = max(abs(sm[i][j] - full[i][j]) for i in range(n) for j in range(n))

    lines = []
    lines.append("Sherman-Morrison-Woodbury: low-rank inverse updates")
    lines.append("=" * 54)
    lines.append(f"n = {n}; update A -> A + u v^T (rank 1)")
    lines.append("")
    lines.append("(A + u v^T)^-1 = A^-1 - (A^-1 u v^T A^-1) / (1 + v^T A^-1 u)")
    lines.append(f"  max error vs full re-inversion: {err:.2e}")
    lines.append("")
    lines.append(f"cost of a rank-1 inverse update: O(n^2) = ~{n*n} flops")
    lines.append(f"cost of a full re-inversion:     O(n^3) = ~{n*n*n} flops")
    lines.append(f"  speedup factor at n={n}: {n}x  (grows with n)")
    lines.append("")
    # determinant lemma
    dl = SM.determinant_lemma(A, u, v)
    lines.append(f"determinant lemma: det(A+uv^T) = {dl:.4f}, direct = {determinant(mod):.4f}")
    lines.append("")
    # Woodbury rank-k timing intuition
    lines.append("Woodbury rank-k: (A + U C V^T)^-1 turns an n x n re-inversion into a k x k one.")
    k = 3
    U = [[rnd() for _ in range(k)] for _ in range(n)]
    V = [[rnd() for _ in range(k)] for _ in range(n)]
    C = [[rnd() * 2 - 1 + (3 if i == j else 0) for j in range(k)] for i in range(k)]
    CVt = [[sum(C[a][bb] * V[j][bb] for bb in range(k)) for j in range(n)] for a in range(k)]
    UCVt = [[sum(U[i][a] * CVt[a][j] for a in range(k)) for j in range(n)] for i in range(n)]
    mod2 = [[A[i][j] + UCVt[i][j] for j in range(n)] for i in range(n)]
    wb = SM.woodbury_inverse(A_inv, U, C, V)
    full2 = inverse(mod2)
    err2 = max(abs(wb[i][j] - full2[i][j]) for i in range(n) for j in range(n))
    lines.append(f"  rank-{k} Woodbury error vs full: {err2:.2e}  "
                 f"(inverts a {k}x{k} instead of an {n}x{n})")
    lines.append("")
    lines.append("These updates power Kalman filters, recursive least squares, and quasi-Newton")
    lines.append("optimizers, where the matrix changes by a low-rank amount every step.")

    text = "\n".join(lines)
    print(text)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        # SVG: op-count curves n^2 (update) vs n^3 (re-inversion) on log-log
        import math
        W, H = 640, 380
        ml, mt, w, h = 60, 55, 520, 250
        ns = [4, 8, 16, 32, 64, 128, 256]
        upd = [nn * nn for nn in ns]
        full_c = [nn ** 3 for nn in ns]
        lx = [math.log10(nn) for nn in ns]
        ally = [math.log10(x) for x in upd + full_c]
        xmin, xmax = min(lx), max(lx)
        ymin, ymax = min(ally), max(ally)
        yr = ymax - ymin

        def sx(v):
            return ml + (v - xmin) / (xmax - xmin) * w

        def sy(v):
            return mt + h - (v - ymin) / yr * h

        s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
        s.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
        s.append(f'<text x="{ml}" y="30" fill="{TEXT}" font-size="15">'
                 f'Low-rank update O(n^2) vs full re-inversion O(n^3)</text>')
        s.append(f'<rect x="{ml}" y="{mt}" width="{w}" height="{h}" fill="none" '
                 f'stroke="{GRAY}" stroke-width="0.6"/>')
        for series, col, lab in ((full_c, YELLOW, "full re-inversion  O(n^3)"),
                                 (upd, GREEN, "rank-1 update  O(n^2)")):
            d = " ".join(f"{sx(lx[i]):.1f},{sy(math.log10(series[i])):.1f}" for i in range(len(ns)))
            s.append(f'<polyline points="{d}" fill="none" stroke="{col}" stroke-width="2.5"/>')
            for i in range(len(ns)):
                s.append(f'<circle cx="{sx(lx[i]):.1f}" cy="{sy(math.log10(series[i])):.1f}" '
                         f'r="2.5" fill="{col}"/>')
        s.append(f'<text x="{ml+10}" y="{mt+18}" fill="{YELLOW}" font-size="11">'
                 f'full re-inversion  O(n^3)</text>')
        s.append(f'<text x="{ml+10}" y="{mt+34}" fill="{GREEN}" font-size="11">'
                 f'rank-1 update  O(n^2)</text>')
        for i in (0, len(ns) - 1):
            s.append(f'<text x="{sx(lx[i]):.1f}" y="{mt+h+16}" fill="{GRAY}" font-size="9" '
                     f'text-anchor="middle">n={ns[i]}</text>')
        s.append(f'<text x="{ml}" y="{H-14}" fill="{GRAY}" font-size="10">'
                 f'The gap between the two lines is the per-step speedup -- one factor of n, and it '
                 f'widens with n. Both hit the exact same answer (error ~ 1e-16).</text>')
        s.append("</svg>")
        with open(os.path.join(outdir, "sherman_morrison.svg"), "w", encoding="utf-8") as fh:
            fh.write("".join(s))

    return text


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
