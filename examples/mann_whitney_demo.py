"""Mann-Whitney demo: a rank test detecting a shift the t-test misses under outliers, with the null distribution."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import mann_whitney as mw


PALETTE = {
    "bg": "#0d1117", "blue": "#4dabf7", "yellow": "#ffd43b", "red": "#ff6b6b",
    "green": "#06d6a0", "purple": "#b197fc", "gray": "#8b949e", "text": "#e6edf3",
}


def t_test_p(a, b):
    """Two-sample Welch t-test p-value (normal approx), for the outlier-sensitivity comparison."""
    na, nb = len(a), len(b)
    ma, mb = sum(a) / na, sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1)
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return 1.0
    t = (ma - mb) / se
    # normal approximation to the two-sided p
    phi = 0.5 * (1 + math.erf(abs(t) / math.sqrt(2)))
    return 2 * (1 - phi)


def main():
    lines = []
    lines.append("Mann-Whitney U test -- nonparametric, rank-based, outlier-robust")
    lines.append("=" * 64)
    lines.append("")

    # two groups with a real shift
    a = [5.1, 5.5, 4.9, 6.0, 5.3, 5.7, 4.7, 5.9, 5.2, 5.6]
    b = [6.2, 6.8, 6.1, 7.0, 6.5, 6.9, 6.3, 7.1, 6.4, 6.7]
    res = mw.mann_whitney(a, b)
    lines.append("Group A ~5.4, Group B ~6.6 (clear shift):")
    lines.append(f"  U = {res['U']}, effect size P(A>B) = {res['effect_size']:.3f}, "
                 f"p = {res['p_value']:.5f} ({res['method']})")
    lines.append(f"  t-test p = {t_test_p(a, b):.5f}  (both detect the shift)")
    lines.append("")

    # now add an extreme outlier to A -- t-test gets confused, rank test doesn't
    a_out = a + [50.0]
    b_out = b + [6.6]
    res_o = mw.mann_whitney(a_out, b_out)
    lines.append("Add one wild outlier (50.0) to group A:")
    lines.append(f"  Mann-Whitney p = {res_o['p_value']:.5f} ({res_o['method']}) -- still sees B > A")
    lines.append(f"  t-test        p = {t_test_p(a_out, b_out):.5f} -- outlier inflates A's mean, masks the shift")
    lines.append("  The rank test uses only order, so one outlier moves it by a single rank.")
    lines.append("")

    # exact null distribution for a small case
    n1, n2 = 5, 6
    counts = mw._exact_null_counts(n1, n2)
    total = sum(counts)
    lines.append(f"Exact null distribution of U1 for n1={n1}, n2={n2} "
                 f"({total} rank arrangements):")
    lines.append(f"  U ranges 0..{n1*n2}, symmetric about {n1*n2/2}, mean = {n1*n2/2}")
    lines.append(f"  peak count {max(counts)} at U={counts.index(max(counts))}")
    lines.append("")

    # effect-size interpretation
    lines.append("Common-language effect size = P(random A beats random B):")
    for shift in (0.0, 0.5, 1.0, 2.0):
        bb = [x + shift for x in a]
        es = mw.effect_size(a, bb)
        lines.append(f"  B shifted by +{shift}: P(A>B) = {es:.3f}")

    text = "\n".join(lines)
    print(text)

    svg = _svg(a, b, counts, n1, n2)
    return text, svg


def _svg(a, b, counts, n1, n2):
    W, H = 640, 430
    P = PALETTE
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
    parts.append(f'<rect width="{W}" height="{H}" fill="{P["bg"]}"/>')
    parts.append(f'<text x="20" y="26" fill="{P["text"]}" font-size="15">'
                 f'Two groups (top) and the exact U null distribution (bottom)</text>')

    # TOP: the two samples on a shared number line
    allv = a + b
    lo, hi = min(allv), max(allv)
    x0, x1 = 55, 610
    ya, yb = 70, 110

    def px(v):
        return x0 + (v - lo) / (hi - lo) * (x1 - x0)

    parts.append(f'<text x="20" y="{ya + 4}" fill="{P["blue"]}" font-size="11">A</text>')
    parts.append(f'<text x="20" y="{yb + 4}" fill="{P["red"]}" font-size="11">B</text>')
    parts.append(f'<line x1="{x0}" y1="{ya}" x2="{x1}" y2="{ya}" stroke="#30363d" stroke-width="1"/>')
    parts.append(f'<line x1="{x0}" y1="{yb}" x2="{x1}" y2="{yb}" stroke="#30363d" stroke-width="1"/>')
    for v in a:
        parts.append(f'<circle cx="{px(v):.1f}" cy="{ya}" r="4" fill="{P["blue"]}" opacity="0.8"/>')
    for v in b:
        parts.append(f'<circle cx="{px(v):.1f}" cy="{yb}" r="4" fill="{P["red"]}" opacity="0.8"/>')
    parts.append(f'<text x="{x0}" y="{yb + 24}" fill="{P["gray"]}" font-size="10">'
                 f'B sits to the right of A -> P(A&gt;B) small -> significant U</text>')

    # BOTTOM: null distribution histogram
    bx0, bx1, by0, by1 = 55, 610, 180, 370
    n = len(counts)
    cmax = max(counts)
    bw = (bx1 - bx0) / n

    def by(c):
        return by1 - c / cmax * (by1 - by0)

    for u in range(n):
        h = by1 - by(counts[u])
        parts.append(f'<rect x="{bx0 + u * bw + 0.5:.1f}" y="{by(counts[u]):.1f}" '
                     f'width="{bw - 1:.1f}" height="{h:.1f}" fill="{P["purple"]}"/>')
    # mean line
    mean_u = n1 * n2 / 2
    parts.append(f'<line x1="{bx0 + mean_u * bw:.1f}" y1="{by0}" x2="{bx0 + mean_u * bw:.1f}" '
                 f'y2="{by1}" stroke="{P["yellow"]}" stroke-width="1.5" stroke-dasharray="4,3"/>')
    parts.append(f'<line x1="{bx0}" y1="{by1}" x2="{bx1}" y2="{by1}" stroke="{P["gray"]}" stroke-width="1"/>')
    parts.append(f'<text x="{bx0}" y="{by0 - 6}" fill="{P["gray"]}" font-size="10">'
                 f'exact null P(U1) for n1={n1}, n2={n2} -- symmetric about the mean (yellow)</text>')
    for u in (0, n1 * n2 // 2, n1 * n2):
        parts.append(f'<text x="{bx0 + u * bw:.1f}" y="{by1 + 14:.1f}" fill="{P["gray"]}" '
                     f'font-size="9" text-anchor="middle">{u}</text>')

    parts.append(f'<text x="20" y="{H - 10}" fill="{P["gray"]}" font-size="11">'
                 f'a significance test asks how far the observed U lands in the tail of this exact '
                 f'combinatorial null.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
