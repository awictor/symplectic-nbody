"""Multiple-testing demo: the BH step-up line rejecting more than Bonferroni, with FDR control on mixed data."""

import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import benjamini_hochberg as mt


PALETTE = {
    "bg": "#0d1117", "blue": "#4dabf7", "yellow": "#ffd43b", "red": "#ff6b6b",
    "green": "#06d6a0", "purple": "#b197fc", "gray": "#8b949e", "text": "#e6edf3",
}


class _R:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def u(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return (self.s >> 8) / (1 << 24)


def main():
    lines = []
    lines.append("Multiple-testing corrections -- FDR (Benjamini-Hochberg) vs FWER")
    lines.append("=" * 64)
    lines.append("")

    # 100 tests: 15 real effects (small p) + 85 nulls (uniform p)
    rng = _R(3)
    n_signal = 15
    signals = [rng.u() * 0.002 for _ in range(n_signal)]
    nulls = [rng.u() for _ in range(85)]
    allp = signals + nulls
    is_sig = [True] * n_signal + [False] * 85

    alpha = 0.05
    rb = mt.bonferroni(allp, alpha)
    rh = mt.holm(allp, alpha)
    rbh = mt.benjamini_hochberg(allp, alpha)

    lines.append(f"100 hypotheses: {n_signal} true effects + 85 nulls, alpha={alpha}.")
    lines.append("")
    lines.append("   method        rejected   true+   false+   FDR")
    lines.append("   " + "-" * 48)
    for name, r in (("uncorrected", {"reject": [p <= alpha for p in allp]}),
                    ("Bonferroni", rb), ("Holm", rh), ("Benjamini-Hochberg", rbh)):
        rej = [i for i in range(100) if r["reject"][i]]
        tp = sum(1 for i in rej if is_sig[i])
        fp = len(rej) - tp
        fdr = fp / len(rej) if rej else 0.0
        lines.append(f"   {name:18s}  {len(rej):4d}      {tp:4d}    {fp:4d}    {fdr:.3f}")
    lines.append("")
    lines.append("Uncorrected: ~5% of 85 nulls slip through as false positives.")
    lines.append("Bonferroni/Holm: control ANY false positive -> conservative, miss real effects.")
    lines.append("Benjamini-Hochberg: control the FRACTION of false discoveries -> recovers more signals.")
    lines.append("")

    # BH threshold detail
    lines.append(f"BH cutoff: reject the {rbh['k']} smallest p-values (largest p rejected = {rbh['cutoff']:.4f}).")
    lines.append(f"Bonferroni cutoff: {rb['threshold']:.5f} (alpha/m).")

    text = "\n".join(lines)
    print(text)

    svg = _svg(allp, is_sig, rbh, rb, alpha)
    return text, svg


def _svg(allp, is_sig, rbh, rb, alpha):
    W, H = 640, 430
    P = PALETTE
    m = len(allp)
    order = sorted(range(m), key=lambda i: allp[i])
    sorted_p = [allp[i] for i in order]

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
    parts.append(f'<rect width="{W}" height="{H}" fill="{P["bg"]}"/>')
    parts.append(f'<text x="20" y="26" fill="{P["text"]}" font-size="15">'
                 f'Sorted p-values vs the BH step-up line and the Bonferroni cutoff</text>')

    x0, x1, y0, y1 = 55, 610, 55, 360
    # y axis: p-value, zoomed to the low region where the action is
    ymax = 0.08

    def px(rank):
        return x0 + rank / (m - 1) * (x1 - x0)

    def py(p):
        p = min(p, ymax)
        return y1 - p / ymax * (y1 - y0)

    # BH line y = (k/m) alpha
    parts.append(f'<line x1="{px(0):.1f}" y1="{py(alpha / m):.1f}" x2="{px(m - 1):.1f}" '
                 f'y2="{py(alpha):.1f}" stroke="{P["green"]}" stroke-width="1.5"/>')
    parts.append(f'<text x="{px(m - 1) - 120:.1f}" y="{py(alpha) - 6:.1f}" fill="{P["green"]}" '
                 f'font-size="10">BH line (k/m)*alpha</text>')

    # Bonferroni horizontal cutoff
    parts.append(f'<line x1="{x0}" y1="{py(rb["threshold"]):.1f}" x2="{x1}" y2="{py(rb["threshold"]):.1f}" '
                 f'stroke="{P["red"]}" stroke-width="1.2" stroke-dasharray="5,4"/>')
    parts.append(f'<text x="{x0 + 5:.1f}" y="{py(rb["threshold"]) - 4:.1f}" fill="{P["red"]}" '
                 f'font-size="10">Bonferroni alpha/m</text>')

    # sorted p-values, colored by true signal, filled if BH-rejected
    for rank, idx in enumerate(order):
        p = allp[idx]
        rejected = rbh["reject"][idx]
        base = P["yellow"] if is_sig[idx] else P["gray"]
        r = 4 if rejected else 2.5
        parts.append(f'<circle cx="{px(rank):.1f}" cy="{py(p):.1f}" r="{r}" '
                     f'fill="{base}" opacity="{0.95 if rejected else 0.5}"/>')

    # mark BH cutoff rank
    if rbh["k"] > 0:
        parts.append(f'<line x1="{px(rbh["k"] - 1):.1f}" y1="{y0}" x2="{px(rbh["k"] - 1):.1f}" '
                     f'y2="{y1}" stroke="{P["purple"]}" stroke-width="1" stroke-dasharray="2,3"/>')
        parts.append(f'<text x="{px(rbh["k"] - 1) + 4:.1f}" y="{y0 + 12:.1f}" fill="{P["purple"]}" '
                     f'font-size="10">BH cutoff k={rbh["k"]}</text>')

    parts.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="{P["gray"]}" stroke-width="1"/>')
    parts.append(f'<text x="{(x0+x1)/2:.1f}" y="{y1 + 20:.1f}" fill="{P["gray"]}" '
                 f'font-size="11" text-anchor="middle">p-values sorted ascending (rank)</text>')
    parts.append(f'<text x="20" y="{H - 12}" fill="{P["gray"]}" font-size="11">'
                 f'yellow = true effect, gray = null; big filled = BH-rejected. BH follows the sloped '
                 f'line, catching far more than the flat Bonferroni cutoff.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
