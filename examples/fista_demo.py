"""FISTA demo: sparse signal recovery via Lasso, and the O(1/k^2) vs O(1/k) acceleration (SVG)."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

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


def _stem_panel(s, xtrue, xhat, ox, oy, w, h):
    p = len(xtrue)
    lo = min(min(xtrue), min(xhat)) - 0.5
    hi = max(max(xtrue), max(xhat)) + 0.5

    def sx(j):
        return ox + (j + 0.5) / p * w

    def sy(v):
        return oy + h - (v - lo) / (hi - lo) * h

    s.append(f'<text x="{ox}" y="{oy-8}" fill="{TEXT}" font-size="12">'
             f'sparse recovery: true vs Lasso estimate</text>')
    s.append(f'<line x1="{ox}" y1="{sy(0):.1f}" x2="{ox+w}" y2="{sy(0):.1f}" stroke="{GRAY}"/>')
    for j in range(p):
        # true (yellow) and estimate (green) stems
        s.append(f'<line x1="{sx(j)-3:.1f}" y1="{sy(0):.1f}" x2="{sx(j)-3:.1f}" y2="{sy(xtrue[j]):.1f}" '
                 f'stroke="{YELLOW}" stroke-width="3"/>')
        s.append(f'<line x1="{sx(j)+3:.1f}" y1="{sy(0):.1f}" x2="{sx(j)+3:.1f}" y2="{sy(xhat[j]):.1f}" '
                 f'stroke="{GREEN}" stroke-width="3"/>')
    s.append(f'<text x="{ox+w-90}" y="{oy+12}" fill="{YELLOW}" font-size="10">true signal</text>')
    s.append(f'<text x="{ox+w-90}" y="{oy+26}" fill="{GREEN}" font-size="10">Lasso (FISTA)</text>')


def _conv_panel(s, hf, hi, fstar, ox, oy, w, h):
    K = min(len(hf), len(hi), 120)
    ef = [max(hf[k] - fstar, 1e-12) for k in range(K)]
    ei = [max(hi[k] - fstar, 1e-12) for k in range(K)]
    lx = [math.log10(k + 1) for k in range(K)]
    ally = [math.log10(v) for v in ef + ei]
    xmin, xmax = min(lx), max(lx)
    ymin, ymax = min(ally), max(ally)
    yr = ymax - ymin or 1

    def sx(v):
        return ox + (v - xmin) / (xmax - xmin) * w

    def sy(v):
        return oy + (ymax - v) / yr * h

    s.append(f'<text x="{ox}" y="{oy-8}" fill="{TEXT}" font-size="12">'
             f'objective error vs iteration (log-log)</text>')
    s.append(f'<rect x="{ox}" y="{oy}" width="{w}" height="{h}" fill="none" stroke="{GRAY}" '
             f'stroke-width="0.6"/>')
    for series, col, lab in ((ei, RED, "ISTA ~1/k"), (ef, GREEN, "FISTA ~1/k^2")):
        d = " ".join(f"{sx(lx[k]):.1f},{sy(math.log10(series[k])):.1f}" for k in range(K))
        s.append(f'<polyline points="{d}" fill="none" stroke="{col}" stroke-width="2"/>')
    s.append(f'<text x="{ox+10}" y="{oy+16}" fill="{RED}" font-size="10">ISTA  ~ 1/k</text>')
    s.append(f'<text x="{ox+10}" y="{oy+30}" fill="{GREEN}" font-size="10">FISTA ~ 1/k^2</text>')


def main(outdir=None):
    rnd = _lcg(20260913)
    n, p = 60, 16
    A = [[rnd() * 2 - 1 for _ in range(p)] for _ in range(n)]
    xtrue = [0.0] * p
    for j in (1, 5, 9, 13):
        xtrue[j] = round(rnd() * 4 - 2, 2)
    b = [sum(A[i][j] * xtrue[j] for j in range(p)) for i in range(n)]

    lam = 0.4

    def f(x):
        r = [sum(A[i][j] * x[j] for j in range(p)) - b[i] for i in range(n)]
        return 0.5 * sum(v * v for v in r) + lam * sum(abs(v) for v in x)

    def grad(x):
        r = [sum(A[i][j] * x[j] for j in range(p)) - b[i] for i in range(n)]
        return F._matTvec(A, r)

    L = F._spectral_norm_sq(A)
    step = 1.0 / L
    prox = lambda z, t: F.prox_l1(z, t * lam)
    xf, hf = F.fista(f, grad, prox, [0.0] * p, step, max_iter=5000, tol=1e-13)
    xi, hi = F.proximal_gradient(f, grad, prox, [0.0] * p, step, max_iter=5000, tol=1e-13)
    fstar = min(min(hf), min(hi))

    lines = []
    lines.append("FISTA: accelerated proximal gradient (Lasso)")
    lines.append("=" * 52)
    lines.append(f"design matrix {n}x{p}, lambda = {lam}, Lipschitz L = {L:.2f}")
    lines.append("")
    lines.append("sparse recovery (nonzero true coefficients):")
    lines.append(f"{'index':>7}{'true':>9}{'Lasso':>9}")
    for j in range(p):
        if abs(xtrue[j]) > 1e-9 or abs(xf[j]) > 1e-4:
            lines.append(f"{j:>7}{xtrue[j]:>9.2f}{xf[j]:>9.3f}")
    nnz = sum(1 for v in xf if abs(v) > 1e-4)
    lines.append(f"recovered {nnz} nonzeros (true: {sum(1 for v in xtrue if v != 0)})")
    lines.append("")

    def iters_to(hist, eps):
        for k, v in enumerate(hist):
            if v - fstar < eps:
                return k
        return len(hist)

    lines.append("iterations to reach objective error:")
    lines.append(f"{'tolerance':>12}{'ISTA':>8}{'FISTA':>8}{'speedup':>10}")
    for eps in (1e-2, 1e-4, 1e-6, 1e-8):
        ki = iters_to(hi, eps)
        kf = iters_to(hf, eps)
        lines.append(f"{eps:>12.0e}{ki:>8}{kf:>8}{ki/max(kf,1):>9.1f}x")
    lines.append("")
    lines.append("Nesterov momentum turns the O(1/k) ISTA rate into O(1/k^2) for free -- same")
    lines.append("gradient and soft-threshold per step, just evaluated at an extrapolated point.")

    text = "\n".join(lines)
    print(text)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        W, H = 720, 340
        s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
        s.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
        s.append(f'<text x="30" y="26" fill="{TEXT}" font-size="15">'
                 f'FISTA: sparse recovery and the momentum speedup</text>')
        _stem_panel(s, xtrue, xf, 40, 60, 320, 240)
        _conv_panel(s, hf, hi, fstar, 410, 60, 280, 220)
        s.append(f'<text x="40" y="{H-14}" fill="{GRAY}" font-size="10">'
                 f'Left: Lasso zeros the inactive coefficients and recovers the active ones. '
                 f'Right: FISTA (green) beats ISTA (red) by an order of magnitude at tight tolerance.</text>')
        s.append("</svg>")
        with open(os.path.join(outdir, "fista.svg"), "w", encoding="utf-8") as fh:
            fh.write("".join(s))

    return text


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
