"""Block bootstrap demo: correct standard errors for autocorrelated data where the i.i.d. bootstrap fails."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import block_bootstrap as bb
import bootstrap as bs


PALETTE = {
    "bg": "#0d1117", "blue": "#4dabf7", "yellow": "#ffd43b", "red": "#ff6b6b",
    "green": "#06d6a0", "purple": "#b197fc", "gray": "#8b949e", "text": "#e6edf3",
}


class _R:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF
        self._spare = None

    def u(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return (self.s >> 8) / (1 << 24)

    def normal(self):
        if self._spare is not None:
            v = self._spare
            self._spare = None
            return v
        u1 = max(self.u(), 1e-12)
        u2 = self.u()
        r = math.sqrt(-2 * math.log(u1))
        self._spare = r * math.sin(2 * math.pi * u2)
        return r * math.cos(2 * math.pi * u2)


def ar1(n, phi, rng):
    x = [0.0]
    for _ in range(n + 200):
        x.append(phi * x[-1] + rng.normal())
    return x[201:201 + n]


def main():
    lines = []
    lines.append("Block bootstrap -- honest standard errors for correlated (time-series) data")
    lines.append("=" * 76)
    lines.append("")

    n = 400
    rng = _R(5)
    series = ar1(n, phi=0.8, rng=rng)

    # analytic long-run SE of the mean for AR(1)
    phi = 0.8
    sigma2 = 1.0 / (1 - phi ** 2)
    lr_se = math.sqrt(sigma2 / n * (1 + phi) / (1 - phi))

    se_iid = bs.bootstrap_se(series, bs.mean, n_resamples=1000, seed=1)
    lines.append(f"AR(1) series, phi={phi}, n={n}. Points are correlated: today ~ yesterday.")
    lines.append("")
    lines.append(f"Analytic long-run SE of the mean:  {lr_se:.4f}")
    lines.append(f"Naive i.i.d. bootstrap SE:         {se_iid:.4f}  ({lr_se/se_iid:.1f}x too small!)")
    lines.append("")
    lines.append("Block bootstrap SE vs block length L (moving-block):")
    lines.append("   L     block SE     vs analytic")
    lines.append("   " + "-" * 34)
    ses = []
    Ls = [1, 2, 5, 10, 20, 40, 80]
    for L in Ls:
        se = bb.block_bootstrap_se(series, bb.mean, block_len=L, n_resamples=1000, seed=1)
        ses.append(se)
        tag = "  <- rule n^(1/3)" if L == bb.optimal_block_length(n) else ""
        lines.append(f"   {L:3d}   {se:.4f}      {se/lr_se:.2f}x{tag}")
    lines.append("")
    lines.append("L=1 IS the i.i.d. bootstrap (too small); as L grows the block SE rises to the truth.")
    lines.append("")

    # variant comparison at a good block length
    L = 20
    lines.append(f"The three variants at L={L}:")
    for method in ("moving", "circular", "stationary"):
        se = bb.block_bootstrap_se(series, bb.mean, block_len=L, n_resamples=1000, method=method, seed=1)
        lines.append(f"   {method:11s}: SE = {se:.4f}")

    text = "\n".join(lines)
    print(text)

    svg = _svg(series, Ls, ses, se_iid, lr_se)
    return text, svg


def _svg(series, Ls, ses, se_iid, lr_se):
    W, H = 640, 450
    P = PALETTE
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
    parts.append(f'<rect width="{W}" height="{H}" fill="{P["bg"]}"/>')
    parts.append(f'<text x="20" y="24" fill="{P["text"]}" font-size="15">'
                 f'AR(1) series (top) and block-bootstrap SE converging to the truth (bottom)</text>')

    # TOP: the correlated series
    n = len(series)
    tx0, tx1, ty0, ty1 = 50, 615, 50, 160
    vmin, vmax = min(series), max(series)

    def tx(i):
        return tx0 + i / (n - 1) * (tx1 - tx0)

    def ty(v):
        return ty1 - (v - vmin) / (vmax - vmin) * (ty1 - ty0)

    pts = " ".join(f"{tx(i):.1f},{ty(series[i]):.1f}" for i in range(n))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="{P["blue"]}" stroke-width="0.8" opacity="0.8"/>')
    parts.append(f'<text x="{tx0}" y="{ty0 - 4:.1f}" fill="{P["gray"]}" font-size="10">'
                 f'smooth wandering = strong positive autocorrelation</text>')

    # BOTTOM: SE vs L
    bx0, bx1, by0, by1 = 55, 610, 230, 400
    smax = max(max(ses), lr_se) * 1.15

    def bx(idx):
        return bx0 + idx / (len(Ls) - 1) * (bx1 - bx0)

    def by(s):
        return by1 - s / smax * (by1 - by0)

    # analytic truth line
    parts.append(f'<line x1="{bx0}" y1="{by(lr_se):.1f}" x2="{bx1}" y2="{by(lr_se):.1f}" '
                 f'stroke="{P["green"]}" stroke-width="1.5" stroke-dasharray="5,4"/>')
    parts.append(f'<text x="{bx1 - 150}" y="{by(lr_se) - 6:.1f}" fill="{P["green"]}" font-size="10">'
                 f'analytic long-run SE</text>')
    # iid SE line
    parts.append(f'<line x1="{bx0}" y1="{by(se_iid):.1f}" x2="{bx1}" y2="{by(se_iid):.1f}" '
                 f'stroke="{P["red"]}" stroke-width="1.2" stroke-dasharray="3,3"/>')
    parts.append(f'<text x="{bx0 + 5:.1f}" y="{by(se_iid) + 12:.1f}" fill="{P["red"]}" font-size="10">'
                 f'i.i.d. bootstrap SE (too small)</text>')
    # block SE curve
    curve = " ".join(f"{bx(i):.1f},{by(ses[i]):.1f}" for i in range(len(Ls)))
    parts.append(f'<polyline points="{curve}" fill="none" stroke="{P["yellow"]}" stroke-width="2"/>')
    for i in range(len(Ls)):
        parts.append(f'<circle cx="{bx(i):.1f}" cy="{by(ses[i]):.1f}" r="3.5" fill="{P["blue"]}"/>')
        parts.append(f'<text x="{bx(i):.1f}" y="{by1 + 14:.1f}" fill="{P["gray"]}" '
                     f'font-size="9" text-anchor="middle">{Ls[i]}</text>')
    parts.append(f'<line x1="{bx0}" y1="{by1}" x2="{bx1}" y2="{by1}" stroke="{P["gray"]}" stroke-width="1"/>')
    parts.append(f'<text x="{(bx0+bx1)/2:.1f}" y="{by1 + 32:.1f}" fill="{P["gray"]}" '
                 f'font-size="11" text-anchor="middle">block length L</text>')

    parts.append(f'<text x="20" y="{H - 12}" fill="{P["gray"]}" font-size="11">'
                 f'the block SE (yellow) rises from the wrong i.i.d. value at L=1 to the true long-run SE '
                 f'as blocks capture the correlation.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
