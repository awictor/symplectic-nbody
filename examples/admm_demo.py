"""ADMM demo: LASSO by operator splitting, showing primal/dual residuals driving to zero (SVG)."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import admm as AD
import fista as F


BG = "#0d1117"
TEXT = "#e6edf3"
GRAY = "#8b949e"
BLUE = "#4dabf7"
GREEN = "#06d6a0"
RED = "#ff6b6b"
YELLOW = "#ffd43b"


def _lcg(seed):
    st = seed & 0xFFFFFFFF

    def rnd():
        nonlocal st
        st = (1664525 * st + 1013904223) & 0xFFFFFFFF
        return (st >> 8) / (1 << 24)

    return rnd


def _stem(s, xtrue, xhat, ox, oy, w, h):
    p = len(xtrue)
    lo = min(min(xtrue), min(xhat)) - 0.5
    hi = max(max(xtrue), max(xhat)) + 0.5

    def sx(j):
        return ox + (j + 0.5) / p * w

    def sy(v):
        return oy + h - (v - lo) / (hi - lo) * h

    s.append(f'<text x="{ox}" y="{oy-8}" fill="{TEXT}" font-size="12">'
             f'LASSO sparse recovery (true vs ADMM)</text>')
    s.append(f'<line x1="{ox}" y1="{sy(0):.1f}" x2="{ox+w}" y2="{sy(0):.1f}" stroke="{GRAY}"/>')
    for j in range(p):
        s.append(f'<line x1="{sx(j)-3:.1f}" y1="{sy(0):.1f}" x2="{sx(j)-3:.1f}" y2="{sy(xtrue[j]):.1f}" '
                 f'stroke="{YELLOW}" stroke-width="3"/>')
        s.append(f'<line x1="{sx(j)+3:.1f}" y1="{sy(0):.1f}" x2="{sx(j)+3:.1f}" y2="{sy(xhat[j]):.1f}" '
                 f'stroke="{GREEN}" stroke-width="3"/>')
    s.append(f'<text x="{ox+w-80}" y="{oy+12}" fill="{YELLOW}" font-size="10">true</text>')
    s.append(f'<text x="{ox+w-80}" y="{oy+26}" fill="{GREEN}" font-size="10">ADMM</text>')


def _residuals(s, hist, ox, oy, w, h):
    K = len(hist)
    prim = [max(hist[k][0], 1e-14) for k in range(K)]
    dual = [max(hist[k][1], 1e-14) for k in range(K)]
    lx = [math.log10(k + 1) for k in range(K)]
    ally = [math.log10(v) for v in prim + dual]
    xmin, xmax = min(lx), max(lx)
    ymin, ymax = min(ally), max(ally)
    yr = ymax - ymin or 1

    def sx(v):
        return ox + (v - xmin) / (xmax - xmin) * w

    def sy(v):
        return oy + (ymax - v) / yr * h

    s.append(f'<text x="{ox}" y="{oy-8}" fill="{TEXT}" font-size="12">'
             f'primal &amp; dual residuals (log-log)</text>')
    s.append(f'<rect x="{ox}" y="{oy}" width="{w}" height="{h}" fill="none" stroke="{GRAY}" '
             f'stroke-width="0.6"/>')
    for series, col, lab in ((prim, BLUE, "primal ||x-z||"), (dual, RED, "dual rho||z-z'||")):
        d = " ".join(f"{sx(lx[k]):.1f},{sy(math.log10(series[k])):.1f}" for k in range(K))
        s.append(f'<polyline points="{d}" fill="none" stroke="{col}" stroke-width="2"/>')
    s.append(f'<text x="{ox+10}" y="{oy+16}" fill="{BLUE}" font-size="10">primal ||x - z||</text>')
    s.append(f'<text x="{ox+10}" y="{oy+30}" fill="{RED}" font-size="10">dual rho||z - z_old||</text>')


def main(outdir=None):
    rnd = _lcg(20260913)
    n, p = 70, 16
    A = [[rnd() * 2 - 1 for _ in range(p)] for _ in range(n)]
    xtrue = [0.0] * p
    for j in (2, 6, 10, 14):
        xtrue[j] = round(rnd() * 4 - 2, 2)
    b = [sum(A[i][j] * xtrue[j] for j in range(p)) for i in range(n)]

    lam = 0.5
    xa, hist = AD.lasso(A, b, lam, rho=1.0, max_iter=3000, tol=1e-10)
    xf = F.lasso_fista(A, b, lam, max_iter=10000)

    lines = []
    lines.append("ADMM: LASSO by operator splitting")
    lines.append("=" * 50)
    lines.append(f"design {n}x{p}, lambda = {lam}, penalty rho = 1.0")
    lines.append("split: f = (1/2)||Ax-b||^2 (linear solve),  g = lam||x||_1 (soft-threshold)")
    lines.append("")
    lines.append("sparse recovery:")
    lines.append(f"{'index':>7}{'true':>9}{'ADMM':>9}{'FISTA':>9}")
    for j in range(p):
        if abs(xtrue[j]) > 1e-9 or abs(xa[j]) > 1e-4:
            lines.append(f"{j:>7}{xtrue[j]:>9.2f}{xa[j]:>9.3f}{xf[j]:>9.3f}")
    lines.append(f"cross-check vs FISTA: max diff = "
                 f"{max(abs(xa[j]-xf[j]) for j in range(p)):.2e}")
    lines.append("")
    lines.append("residual history (converges when both -> 0):")
    lines.append(f"{'iter':>6}{'primal ||x-z||':>18}{'dual rho||dz||':>18}")
    marks = [0, 4, 9, 19, len(hist) - 1]
    for k in marks:
        if k < len(hist):
            lines.append(f"{k:>6}{hist[k][0]:>18.2e}{hist[k][1]:>18.2e}")
    lines.append(f"converged in {len(hist)} iterations")
    lines.append("")
    lines.append("ADMM decouples the smooth fit (one cached matrix factorization) from the")
    lines.append("non-smooth L1 penalty (element-wise soft-threshold), linked by the dual variable.")

    text = "\n".join(lines)
    print(text)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        W, H = 720, 340
        s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
        s.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
        s.append(f'<text x="30" y="26" fill="{TEXT}" font-size="15">'
                 f'ADMM: split a coupled problem into easy prox steps</text>')
        _stem(s, xtrue, xa, 40, 60, 320, 240)
        _residuals(s, hist, 410, 60, 280, 220)
        s.append(f'<text x="40" y="{H-14}" fill="{GRAY}" font-size="10">'
                 f'Left: the recovered signal (green) matches the planted spikes (yellow). '
                 f'Right: primal and dual residuals both fall to zero -- the constraint x = z is met.</text>')
        s.append("</svg>")
        with open(os.path.join(outdir, "admm.svg"), "w", encoding="utf-8") as fh:
            fh.write("".join(s))

    return text


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
