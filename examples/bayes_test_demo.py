"""Demo: Bayes and the base-rate fallacy.

Prints the famous rare-disease posterior against a Monte-Carlo cohort, then draws how the
positive predictive value climbs with prevalence (crossing 50% only at the base-rate threshold)
and a breakdown of a 100,000-person cohort showing false positives swamping the true ones.

    python examples/bayes_test_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bayes_test import (posterior_positive, posterior_after_retests,  # noqa: E402
                        prevalence_for_even_odds, positive_likelihood_ratio, simulate)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    pr, se, sp = 0.001, 0.99, 0.99
    print("Bayes & the base-rate fallacy: a 99%-accurate test, a 1-in-1000 disease\n")
    ppv = posterior_positive(pr, se, sp)
    print(f"  prevalence           = {pr:.3%}")
    print(f"  sensitivity          = {se:.0%}   specificity = {sp:.0%}")
    print(f"  P(sick | positive)   = {ppv:.1%}   <- not 99%!   (Monte-Carlo: {simulate(pr, se, sp, n=1000000, seed=7):.1%})")
    print(f"  P(sick | 2 positives)= {posterior_after_retests(pr, se, sp, 2):.1%}")
    print(f"  break-even prevalence= {prevalence_for_even_odds(se, sp):.2%}  (a positive is 50-50 here)")

    print("\n  In 100,000 people: 100 are sick and 99 test positive, but 99,900 are healthy and")
    print("  1% of them -- 999 -- test positive too. So 99 of 1098 positives are real: 9%. The")
    print("  rare disease lets false positives swamp true ones however good the test sounds --")
    print("  the base-rate fallacy behind medical screening, spam filters, and profiling.")

    _svg(os.path.join(outdir, "bayes_test.svg"), pr, se, sp)
    print(f"\n  wrote {os.path.join(outdir, 'bayes_test.svg')}")


def _svg(path, pr, se, sp, w=760, h=380):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="monospace">',
        f'<rect width="{w}" height="{h}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Bayes: a positive test on a rare disease is usually a false alarm</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'positive predictive value vs prevalence (left); a 100,000-person cohort (right)</text>',
    ]

    # left: PPV vs prevalence (log x), crossing 50% at the base-rate threshold
    lx0, lx1 = 60, w // 2 - 20
    ly0, ly1 = h - 55, 62
    prevs = [10 ** (-4 + 4 * i / 200) for i in range(201)]  # 1e-4 .. 1

    def LX(pv):
        return lx0 + (math.log10(pv) + 4) / 4 * (lx1 - lx0)

    def LY(v):
        return ly0 - v * (ly0 - ly1)

    parts.append(f'<line x1="{lx0}" y1="{ly0}" x2="{lx1}" y2="{ly0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{lx0}" y1="{ly0}" x2="{lx0}" y2="{ly1}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{lx0}" y1="{LY(0.5):.1f}" x2="{lx1}" y2="{LY(0.5):.1f}" '
                 f'stroke="#21262d" stroke-width="1"/>')
    curve = " ".join(f"{LX(pv):.1f},{LY(posterior_positive(pv, se, sp)):.1f}" for pv in prevs)
    parts.append(f'<polyline points="{curve}" fill="none" stroke="#4dabf7" stroke-width="2.5"/>')
    # threshold and the classic 0.1% point
    pstar = prevalence_for_even_odds(se, sp)
    parts.append(f'<line x1="{LX(pstar):.1f}" y1="{ly1}" x2="{LX(pstar):.1f}" y2="{ly0}" '
                 f'stroke="#06d6a0" stroke-width="1" stroke-dasharray="3 3"/>')
    parts.append(f'<circle cx="{LX(pstar):.1f}" cy="{LY(0.5):.1f}" r="3.5" fill="#06d6a0"/>')
    parts.append(f'<text x="{LX(pstar)+5:.1f}" y="{LY(0.5)-6:.1f}" fill="#06d6a0" font-size="9">'
                 f'50-50 at {pstar:.0%}</text>')
    parts.append(f'<circle cx="{LX(pr):.1f}" cy="{LY(posterior_positive(pr, se, sp)):.1f}" r="3.5" fill="#ff6b6b"/>')
    parts.append(f'<text x="{LX(pr)+5:.1f}" y="{LY(posterior_positive(pr, se, sp))+4:.1f}" '
                 f'fill="#ff6b6b" font-size="9">1-in-1000 -> {posterior_positive(pr, se, sp):.0%}</text>')
    for v in (0.0, 0.5, 1.0):
        parts.append(f'<text x="{lx0-6:.1f}" y="{LY(v)+3:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="end">{v:.1f}</text>')
    for e in (-4, -3, -2, -1, 0):
        parts.append(f'<text x="{LX(10**e):.1f}" y="{ly0+15:.1f}" fill="#8b949e" font-size="8" '
                     f'text-anchor="middle">{10**e:g}</text>')
    parts.append(f'<text x="{(lx0+lx1)/2:.1f}" y="{ly0+30:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">prevalence (log)</text>')

    # right: 100k cohort stacked bar -- true positives vs false positives among all positives
    rx0 = w // 2 + 45
    ry0, ry1 = h - 55, 70
    N = 100000
    sick = N * pr
    tp = sick * se
    fp = (N - sick) * (1.0 - sp)
    total_pos = tp + fp
    barw = 90
    barh = ry0 - ry1
    bx = rx0 + 40

    def seg(y0, val, tot, col, label):
        hh = barh * val / tot
        parts.append(f'<rect x="{bx}" y="{y0 - hh:.1f}" width="{barw}" height="{hh:.1f}" '
                     f'fill="{col}"/>')
        parts.append(f'<text x="{bx + barw + 8}" y="{y0 - hh/2 + 4:.1f}" fill="{col}" '
                     f'font-size="10">{label}: {int(round(val))}</text>')
        return y0 - hh

    y = ry0
    y = seg(y, fp, total_pos, "#ff6b6b", "false positives")
    y = seg(y, tp, total_pos, "#06d6a0", "true positives")
    parts.append(f'<text x="{bx + barw/2:.1f}" y="{ry0+16:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">the {int(round(total_pos))} positives</text>')
    parts.append(f'<text x="{bx + barw/2:.1f}" y="{ry1-8:.1f}" fill="#e6edf3" font-size="11" '
                 f'text-anchor="middle">only {tp/total_pos:.0%} real</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
