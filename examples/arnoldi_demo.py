"""Arnoldi demo: pull the dominant eigenvalues of a large non-symmetric matrix from a tiny subspace, and watch the Ritz values converge."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import arnoldi
import qr_algorithm


PALETTE = {
    "bg": "#0d1117", "blue": "#4dabf7", "yellow": "#ffd43b", "red": "#ff6b6b",
    "green": "#06d6a0", "purple": "#b197fc", "gray": "#8b949e", "text": "#e6edf3",
}


class _R:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def u(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return (self.s >> 8) / (1 << 24) * 2 - 1


def rand_matrix(n, seed):
    r = _R(seed)
    return [[r.u() for _ in range(n)] for _ in range(n)]


def main():
    lines = []
    lines.append("Arnoldi iteration -- dominant eigenvalues from a Krylov subspace")
    lines.append("=" * 64)
    lines.append("")

    n = 40
    A = rand_matrix(n, 42)
    # boost one diagonal entry for a clean dominant real mode
    A[0][0] += 22.0
    mv = arnoldi.matvec_from_matrix(A)

    true = qr_algorithm.eigenvalues(A)
    true.sort(key=lambda z: -abs(z))
    lam_true = true[0]
    lines.append(f"{n}x{n} non-symmetric matrix. True dominant eigenvalue: {lam_true.real:.6f}")
    lines.append(f"(computed by full O(n^3) QR-algorithm on all {n} rows)")
    lines.append("")

    lines.append("Arnoldi from a random start vector, matrix-vector products only:")
    lines.append("   m    dominant Ritz value      |error|      matvecs")
    lines.append("  " + "-" * 52)
    conv = []
    for m in (2, 4, 6, 8, 10, 14, 18):
        r = arnoldi.ritz_values(mv, n, m=m, seed=3)
        err = abs(r[0] - lam_true)
        conv.append((m, err))
        lines.append(f"  {m:3d}    {r[0].real:16.9f}    {err:9.2e}    {m:5d}")
    lines.append("")
    lines.append(f"Full solve touches all {n} rows; Arnoldi nails the dominant mode in")
    lines.append(f"~10 matvecs -- {n * n} multiplies vs {10 * n}, the Krylov advantage.")
    lines.append("")

    # dominant Ritz value at moderate m vs true; interior modes need larger m
    m = 12
    ritz = arnoldi.ritz_values(mv, n, m=m, seed=3)
    lines.append(f"Arnoldi m={m}: the dominant mode is locked, interior Ritz values still")
    lines.append("drift (extremal eigenvalues converge first -- that's the whole point):")
    lines.append(f"  dominant Ritz  {ritz[0].real:+.6f}   true  {true[0].real:+.6f}   "
                 f"(|err| {abs(ritz[0] - true[0]):.1e})")

    text = "\n".join(lines)
    print(text)

    svg = _svg(conv, true, ritz)
    return text, svg


def _svg(conv, true, ritz):
    W, H = 640, 420
    P = PALETTE
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
    parts.append(f'<rect width="{W}" height="{H}" fill="{P["bg"]}"/>')
    parts.append(f'<text x="20" y="26" fill="{P["text"]}" font-size="15">'
                 f'Arnoldi: Ritz convergence + spectrum in the complex plane</text>')

    # LEFT PANEL: log-error convergence curve
    lx0, lx1, ly0, ly1 = 55, 300, 60, 360
    ms = [c[0] for c in conv]
    errs = [max(c[1], 1e-16) for c in conv]
    logs = [math.log10(e) for e in errs]
    emin, emax = min(logs), max(logs)
    mmin, mmax = min(ms), max(ms)

    def px(m):
        return lx0 + (m - mmin) / (mmax - mmin) * (lx1 - lx0)

    def py(le):
        return ly1 - (le - emin) / (emax - emin) * (ly1 - ly0)

    # axes
    parts.append(f'<line x1="{lx0}" y1="{ly0}" x2="{lx0}" y2="{ly1}" stroke="{P["gray"]}" stroke-width="1"/>')
    parts.append(f'<line x1="{lx0}" y1="{ly1}" x2="{lx1}" y2="{ly1}" stroke="{P["gray"]}" stroke-width="1"/>')
    parts.append(f'<text x="{lx0}" y="{ly0 - 6}" fill="{P["gray"]}" font-size="10">log10 |error|</text>')
    parts.append(f'<text x="{lx1 - 60}" y="{ly1 + 18}" fill="{P["gray"]}" font-size="10">subspace m</text>')
    pts = " ".join(f"{px(ms[i]):.1f},{py(logs[i]):.1f}" for i in range(len(ms)))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="{P["yellow"]}" stroke-width="2"/>')
    for i in range(len(ms)):
        parts.append(f'<circle cx="{px(ms[i]):.1f}" cy="{py(logs[i]):.1f}" r="3.5" fill="{P["blue"]}"/>')
    parts.append(f'<text x="{lx0 + 6}" y="{ly1 + 18}" fill="{P["gray"]}" font-size="9">'
                 f'error falls ~exponentially in m</text>')

    # RIGHT PANEL: complex-plane spectrum (true vs Ritz)
    rx0, rx1, ry0, ry1 = 360, 610, 60, 360
    reals = [z.real for z in true] + [z.real for z in ritz]
    imags = [z.imag for z in true] + [z.imag for z in ritz]
    rmin, rmax = min(reals), max(reals)
    imin, imax = min(imags), max(imags)
    pad = 0.15 * max(rmax - rmin, 1e-9)
    rmin -= pad
    rmax += pad
    ipad = 0.15 * max(imax - imin, 1e-9)
    imin -= ipad
    imax += ipad

    def cx(re):
        return rx0 + (re - rmin) / (rmax - rmin) * (rx1 - rx0)

    def cy(im):
        return ry1 - (im - imin) / (imax - imin) * (ry1 - ry0)

    # zero axes
    if rmin < 0 < rmax:
        parts.append(f'<line x1="{cx(0):.1f}" y1="{ry0}" x2="{cx(0):.1f}" y2="{ry1}" stroke="#333" stroke-width="1"/>')
    if imin < 0 < imax:
        parts.append(f'<line x1="{rx0}" y1="{cy(0):.1f}" x2="{rx1}" y2="{cy(0):.1f}" stroke="#333" stroke-width="1"/>')
    # true eigenvalues (hollow gray)
    for z in true:
        parts.append(f'<circle cx="{cx(z.real):.1f}" cy="{cy(z.imag):.1f}" r="4" '
                     f'fill="none" stroke="{P["gray"]}" stroke-width="1.2"/>')
    # Ritz values (filled green/purple: real vs complex)
    for z in ritz:
        col = P["green"] if abs(z.imag) < 1e-6 else P["purple"]
        parts.append(f'<circle cx="{cx(z.real):.1f}" cy="{cy(z.imag):.1f}" r="2.6" fill="{col}"/>')
    parts.append(f'<text x="{rx0}" y="{ry0 - 6}" fill="{P["gray"]}" font-size="10">'
                 f'complex plane: gray=true, green/purple=Ritz</text>')
    parts.append(f'<text x="{rx0}" y="{ry1 + 18}" fill="{P["gray"]}" font-size="10">Re</text>')

    parts.append(f'<text x="20" y="{H - 12}" fill="{P["gray"]}" font-size="11">'
                 f'The few extremal Ritz values lock onto the true dominant eigenvalues; interior ones lag.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
