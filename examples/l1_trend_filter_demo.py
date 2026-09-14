"""L1 trend filter demo: fit a piecewise-linear trend to noisy data, auto-locate the kinks, and compare to Hodrick-Prescott."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import l1_trend_filter as l1
import hp_filter


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


def main():
    lines = []
    lines.append("L1 trend filtering -- piecewise-linear trend with automatic knots")
    lines.append("=" * 66)
    lines.append("")

    # clean piecewise-linear ground truth: up, plateau, down, up
    n = 100
    true = []
    val = 0.0
    breaks = [25, 50, 75]
    slopes = [0.8, 0.0, -0.6, 0.4]
    for i in range(n):
        seg = 0
        for b in breaks:
            if i >= b:
                seg += 1
        val = val + (slopes[seg] if i > 0 else 0.0)
        true.append(val)

    rng = _R(11)
    noisy = [true[i] + 1.5 * rng.u() for i in range(n)]

    fit = l1.l1_trend_filter(noisy, lam=18.0, max_iter=8000)
    ks = l1.kinks(fit, tol=3e-2)

    rmse_fit = math.sqrt(sum((fit[i] - true[i]) ** 2 for i in range(n)) / n)
    rmse_noise = math.sqrt(sum((noisy[i] - true[i]) ** 2 for i in range(n)) / n)

    lines.append(f"Signal: piecewise-linear, {len(breaks)} true breakpoints at {breaks}.")
    lines.append(f"Noisy data RMSE from truth: {rmse_noise:.3f}")
    lines.append(f"L1 fit    RMSE from truth: {rmse_fit:.3f}   ({rmse_noise / rmse_fit:.1f}x cleaner)")
    lines.append(f"Detected kinks: {ks}")
    lines.append("")

    lines.append("lambda sweep -- larger lambda, fewer segments:")
    lines.append("   lambda   #kinks   RMSE-from-truth")
    lines.append("   " + "-" * 34)
    for lam in (0.1, 1.0, 4.0, 20.0, 100.0):
        x = l1.l1_trend_filter(noisy, lam=lam, max_iter=6000)
        rk = len(l1.kinks(x, tol=1e-2))
        rm = math.sqrt(sum((x[i] - true[i]) ** 2 for i in range(n)) / n)
        lines.append(f"   {lam:6g}    {rk:4d}     {rm:.3f}")
    lines.append("")

    # vs Hodrick-Prescott: same data, L2 penalty -> smoothly curved, corners smeared
    hp, _cycle = hp_filter.hp_filter(noisy, lam=200.0)
    d2_l1 = l1.second_difference(fit)
    d2_hp = l1.second_difference(hp)
    nnz_l1 = sum(1 for v in d2_l1 if abs(v) > 1e-2)
    nnz_hp = sum(1 for v in d2_hp if abs(v) > 1e-2)
    lines.append("vs Hodrick-Prescott (L2 penalty on the same data):")
    lines.append(f"  L1 second-difference nonzeros: {nnz_l1}  (sparse -> sharp corners)")
    lines.append(f"  HP second-difference nonzeros: {nnz_hp}  (dense -> everywhere curved)")

    text = "\n".join(lines)
    print(text)

    svg = _svg(noisy, true, fit, hp, ks)
    return text, svg


def _svg(noisy, true, fit, hp, ks):
    W, H = 640, 420
    P = PALETTE
    n = len(noisy)
    allv = noisy + true + fit + hp
    ymin, ymax = min(allv), max(allv)
    pad = 0.08 * (ymax - ymin)
    ymin -= pad
    ymax += pad

    x0, x1, y0, y1 = 45, 620, 55, 360

    def px(i):
        return x0 + i / (n - 1) * (x1 - x0)

    def py(v):
        return y1 - (v - ymin) / (ymax - ymin) * (y1 - y0)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
    parts.append(f'<rect width="{W}" height="{H}" fill="{P["bg"]}"/>')
    parts.append(f'<text x="20" y="26" fill="{P["text"]}" font-size="15">'
                 f'L1 trend filter: piecewise-linear fit vs Hodrick-Prescott</text>')

    # kink verticals
    for k in ks:
        parts.append(f'<line x1="{px(k):.1f}" y1="{y0}" x2="{px(k):.1f}" y2="{y1}" '
                     f'stroke="#333" stroke-width="1" stroke-dasharray="3,3"/>')

    # noisy data as faint dots
    for i in range(n):
        parts.append(f'<circle cx="{px(i):.1f}" cy="{py(noisy[i]):.1f}" r="1.7" '
                     f'fill="{P["gray"]}" opacity="0.55"/>')

    # HP trend (purple, smooth)
    hp_pts = " ".join(f"{px(i):.1f},{py(hp[i]):.1f}" for i in range(n))
    parts.append(f'<polyline points="{hp_pts}" fill="none" stroke="{P["purple"]}" '
                 f'stroke-width="1.6" opacity="0.9"/>')

    # L1 fit (yellow, piecewise-linear)
    l1_pts = " ".join(f"{px(i):.1f},{py(fit[i]):.1f}" for i in range(n))
    parts.append(f'<polyline points="{l1_pts}" fill="none" stroke="{P["yellow"]}" stroke-width="2.2"/>')

    # kink markers on the L1 fit
    for k in ks:
        parts.append(f'<circle cx="{px(k):.1f}" cy="{py(fit[k]):.1f}" r="3.5" '
                     f'fill="{P["red"]}"/>')

    # legend
    parts.append(f'<text x="{x0}" y="{y1 + 22}" fill="{P["gray"]}" font-size="11">'
                 f'gray = noisy data</text>')
    parts.append(f'<text x="{x0 + 150}" y="{y1 + 22}" fill="{P["yellow"]}" font-size="11">'
                 f'yellow = L1 (piecewise-linear)</text>')
    parts.append(f'<text x="{x0 + 380}" y="{y1 + 22}" fill="{P["purple"]}" font-size="11">'
                 f'purple = HP (smooth)</text>')
    parts.append(f'<text x="{x0}" y="{y1 + 40}" fill="{P["red"]}" font-size="11">'
                 f'red dots = detected kinks (auto-placed knots)</text>')
    parts.append(f'<text x="20" y="{H - 10}" fill="{P["gray"]}" font-size="11">'
                 f'L1 keeps sharp corners; HP rounds them -- the L1 second difference is zero except at '
                 f'the {len(ks)} kinks.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
