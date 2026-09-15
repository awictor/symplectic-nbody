"""Wilcoxon signed-rank demo: a paired test robust to an outlier that flips the paired t-test, with the null."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import wilcoxon_signed_rank as wsr


PALETTE = {
    "bg": "#0d1117", "blue": "#4dabf7", "yellow": "#ffd43b", "red": "#ff6b6b",
    "green": "#06d6a0", "purple": "#b197fc", "gray": "#8b949e", "text": "#e6edf3",
}


def paired_t_p(diffs):
    """Paired t-test p-value (normal approx) on the differences."""
    n = len(diffs)
    m = sum(diffs) / n
    var = sum((d - m) ** 2 for d in diffs) / (n - 1)
    se = math.sqrt(var / n)
    if se == 0:
        return 0.0
    t = m / se
    phi = 0.5 * (1 + math.erf(abs(t) / math.sqrt(2)))
    return 2 * (1 - phi)


def main():
    lines = []
    lines.append("Wilcoxon signed-rank test -- robust paired comparison")
    lines.append("=" * 54)
    lines.append("")

    # before/after on 12 subjects: a real, consistent improvement
    before = [23.1, 25.4, 22.8, 26.0, 24.3, 21.9, 25.1, 23.7, 24.8, 22.5, 26.3, 23.0]
    after = [24.0, 26.1, 24.0, 26.9, 25.0, 23.5, 25.9, 25.2, 25.6, 24.0, 27.0, 24.3]
    diffs = [after[i] - before[i] for i in range(len(before))]
    res = wsr.wilcoxon(diffs, alternative="greater")
    lines.append("12 subjects, 'after - before' (a consistent improvement):")
    lines.append(f"  differences: {[round(d, 1) for d in diffs]}")
    lines.append(f"  W+ = {res['W_plus']:.1f}, W- = {res['W_minus']:.1f}")
    lines.append(f"  Wilcoxon p (greater) = {res['p_value']:.4f} ({res['method']})")
    lines.append(f"  paired t-test p = {paired_t_p(diffs):.4f}  (agrees here)")
    lines.append("")

    # now one subject has a data-entry error (huge negative outlier)
    diffs_out = list(diffs)
    diffs_out[5] = -50.0
    res_out = wsr.wilcoxon(diffs_out, alternative="greater")
    lines.append("One subject's difference is a data-entry error (-50):")
    lines.append(f"  Wilcoxon p (greater) = {res_out['p_value']:.4f}  -- still sees the improvement")
    lines.append(f"  paired t-test  p     = {paired_t_p(diffs_out):.4f}  -- outlier destroys it")
    lines.append("  The rank test caps the outlier's influence at one rank; the t-test averages it in.")
    lines.append("")

    # exact null distribution
    n = res["n"]
    counts = wsr._exact_null_counts(n)
    total = sum(counts)
    lines.append(f"Exact null distribution of W+ for n={n} (2^{n} = {total} sign patterns):")
    lines.append(f"  W+ ranges 0..{n*(n+1)//2}, symmetric about {n*(n+1)//4}")
    lines.append(f"  observed W+ = {res['W_plus']:.0f} sits in the extreme upper tail")
    lines.append("")

    # power vs effect size
    lines.append("Power vs shift size (15 pairs, unit noise):")
    lines.append("   shift   Wilcoxon p")
    lines.append("   " + "-" * 22)
    for shift in (0.0, 0.3, 0.6, 1.0, 1.5):
        rng_s = _R(9)
        d = [shift + rng_s.normal() for _ in range(15)]
        lines.append(f"   {shift:5.1f}   {wsr.wilcoxon(d)['p_value']:.4f}")

    text = "\n".join(lines)
    print(text)

    svg = _svg(before, after, counts, res, n)
    return text, svg


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


def _svg(before, after, counts, res, n):
    W, H = 640, 450
    P = PALETTE
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
    parts.append(f'<rect width="{W}" height="{H}" fill="{P["bg"]}"/>')
    parts.append(f'<text x="20" y="24" fill="{P["text"]}" font-size="15">'
                 f'Paired before/after (top) and the exact W+ null (bottom)</text>')

    # TOP: paired dumbbell plot
    m = len(before)
    allv = before + after
    vmin, vmax = min(allv), max(allv)
    tx0, tx1, ty0, ty1 = 55, 610, 55, 175

    def px(i):
        return tx0 + i / (m - 1) * (tx1 - tx0)

    def ty(v):
        return ty1 - (v - vmin) / (vmax - vmin) * (ty1 - ty0)

    for i in range(m):
        # connecting line, green if increased
        col = P["green"] if after[i] > before[i] else P["red"]
        parts.append(f'<line x1="{px(i):.1f}" y1="{ty(before[i]):.1f}" x2="{px(i):.1f}" '
                     f'y2="{ty(after[i]):.1f}" stroke="{col}" stroke-width="1.5"/>')
        parts.append(f'<circle cx="{px(i):.1f}" cy="{ty(before[i]):.1f}" r="3" fill="{P["gray"]}"/>')
        parts.append(f'<circle cx="{px(i):.1f}" cy="{ty(after[i]):.1f}" r="3" fill="{P["blue"]}"/>')
    parts.append(f'<text x="{tx0}" y="{ty0 - 4:.1f}" fill="{P["gray"]}" font-size="10">'
                 f'gray=before, blue=after; green line=increase (almost all up)</text>')

    # BOTTOM: exact null of W+
    total = sum(counts)
    bx0, bx1, by0, by1 = 55, 610, 235, 400
    max_w = len(counts) - 1
    cmax = max(counts)
    bw = (bx1 - bx0) / len(counts)

    def by(c):
        return by1 - c / cmax * (by1 - by0)

    def wx(w):
        return bx0 + w / max_w * (bx1 - bx0)

    for w in range(len(counts)):
        h = by1 - by(counts[w])
        # shade the observed tail (>= W+)
        col = P["red"] if w >= res["W_plus"] - 1e-9 else P["purple"]
        parts.append(f'<rect x="{wx(w):.1f}" y="{by(counts[w]):.1f}" width="{max(bw-0.5,1):.1f}" '
                     f'height="{h:.1f}" fill="{col}" opacity="0.85"/>')
    # observed marker
    parts.append(f'<line x1="{wx(res["W_plus"]):.1f}" y1="{by0}" x2="{wx(res["W_plus"]):.1f}" '
                 f'y2="{by1}" stroke="{P["yellow"]}" stroke-width="1.5"/>')
    parts.append(f'<text x="{wx(res["W_plus"]) - 30:.1f}" y="{by0 + 12:.1f}" fill="{P["yellow"]}" '
                 f'font-size="10">observed W+</text>')
    parts.append(f'<line x1="{bx0}" y1="{by1}" x2="{bx1}" y2="{by1}" stroke="{P["gray"]}" stroke-width="1"/>')
    parts.append(f'<text x="{bx0}" y="{by0 - 6:.1f}" fill="{P["gray"]}" font-size="10">'
                 f'exact null of W+ (n={n}); red tail = as-or-more extreme = the p-value</text>')

    parts.append(f'<text x="20" y="{H - 10}" fill="{P["gray"]}" font-size="11">'
                 f'the observed W+ lands in the red upper tail -- a significant paired shift, no '
                 f'normality assumed.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
