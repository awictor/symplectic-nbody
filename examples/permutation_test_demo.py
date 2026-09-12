"""Demo: permutation tests -- significance by shuffling labels, no distributional assumptions.

Runs an exact two-sample test on a small dataset (complete enumeration), a Monte-Carlo test on
larger samples, a paired sign-flip test, and draws the permutation null distribution with the
observed statistic and the rejection region marked.

    python examples/permutation_test_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from permutation_test import (permutation_test, sign_flip_test, diff_of_means,  # noqa: E402
                              t_statistic, mean, _LCG, _shuffle)


class LCG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def u(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return (self.s >> 8) / (1 << 24)

    def normal(self, mu=0.0, sigma=1.0):
        return mu + sigma * (sum(self.u() for _ in range(12)) - 6.0)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Permutation test: is the gap real, or just how the labels fell?\n")

    # small, exact
    treated = [5.1, 6.2, 4.8, 7.0, 5.5]
    control = [3.2, 4.1, 2.9, 3.8]
    r = permutation_test(treated, control, diff_of_means)
    print("  small trial (exact enumeration of all label assignments):")
    print(f"    treated mean {mean(treated):.2f}, control mean {mean(control):.2f}, "
          f"observed diff {r['observed']:.3f}")
    print(f"    enumerated {r['n_permutations']} permutations -> exact p = {r['p_value']:.4f}")
    print(f"    {'significant at 0.05' if r['p_value'] < 0.05 else 'not significant at 0.05'}\n")

    # larger, Monte-Carlo, with t-statistic
    rng = LCG(11)
    ga = [rng.normal(10.0, 3.0) for _ in range(50)]
    gb = [rng.normal(11.8, 3.0) for _ in range(50)]
    rt = permutation_test(ga, gb, t_statistic, n_resamples=20000, seed=5)
    print("  larger trial (Monte-Carlo, t-statistic, 20000 shuffles):")
    print(f"    group A mean {mean(ga):.2f}, group B mean {mean(gb):.2f}")
    print(f"    observed t = {rt['observed']:.3f}, p = {rt['p_value']:.4f} "
          f"({'exact' if rt['exact'] else 'Monte-Carlo'})\n")

    # paired sign-flip
    before = [22.1, 19.8, 25.3, 21.0, 23.7, 20.4, 24.9, 22.8]
    after = [20.3, 18.1, 23.0, 19.9, 21.8, 19.0, 22.7, 21.1]
    diffs = [b - a for b, a in zip(before, after)]
    sr = sign_flip_test(diffs)
    print("  paired before/after (sign-flip, exact over 2**n sign patterns):")
    print(f"    mean drop {sr['observed']:.3f} across {len(diffs)} subjects")
    print(f"    enumerated {sr['n_permutations']} sign patterns -> p = {sr['p_value']:.4f}")
    print(f"    {'significant at 0.05' if sr['p_value'] < 0.05 else 'not significant at 0.05'}\n")

    print("  The p-value IS the fraction of relabellings at least as extreme as what we saw. Under")
    print("  the null the labels are exchangeable, so this fraction is the true tail probability --")
    print("  exact when enumerated, Monte-Carlo estimated with the (b+1)/(B+1) correction otherwise.")

    _svg(os.path.join(outdir, "permutation_test.svg"), ga, gb)
    print(f"\n  wrote {os.path.join(outdir, 'permutation_test.svg')}")


def _svg(path, ga, gb, width=760, height=380):
    # build the Monte-Carlo null distribution of diff-of-means by shuffling
    pool = ga + gb
    na = len(ga)
    observed = mean(ga) - mean(gb)
    rng = _LCG(21)
    stats = []
    for _ in range(6000):
        perm = _shuffle(pool, rng)
        stats.append(mean(perm[:na]) - mean(perm[na:]))

    smin, smax = min(stats), max(stats)
    # make sure the observed value is inside the frame
    smin = min(smin, observed) - 0.1
    smax = max(smax, observed) + 0.1
    span = smax - smin if smax > smin else 1.0
    bins = 45
    counts = [0] * bins
    for s in stats:
        bi = min(bins - 1, int((s - smin) / span * bins))
        counts[bi] += 1
    cmax = max(counts)

    ox, oy = 50, 320
    pw, ph = width - 90, 250

    def px(v):
        return ox + (v - smin) / span * pw

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="17">'
        f'Permutation null distribution (6000 label shuffles)</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'diff-of-means under random relabelling; red bars are at least as extreme as observed</text>',
    ]
    # bars: red if |stat| >= |observed| (contributes to the two-sided p-value)
    bw = pw / bins
    thresh = abs(observed) - 1e-12
    for b in range(bins):
        h = ph * counts[b] / cmax
        x = ox + b * bw
        # centre value of this bin
        cv = smin + (b + 0.5) / bins * span
        colour = "#ff6b6b" if abs(cv) >= thresh else "#4dabf7"
        parts.append(f'<rect x="{x:.1f}" y="{oy-h:.1f}" width="{bw-1:.1f}" height="{h:.1f}" '
                     f'fill="{colour}"/>')
    # observed line
    parts.append(f'<line x1="{px(observed):.1f}" y1="{oy-ph:.1f}" x2="{px(observed):.1f}" '
                 f'y2="{oy:.1f}" stroke="#ffd43b" stroke-width="2"/>')
    parts.append(f'<text x="{px(observed):.0f}" y="{oy-ph-4:.0f}" fill="#ffd43b" font-size="11" '
                 f'text-anchor="middle">observed {observed:.2f}</text>')
    # zero line
    parts.append(f'<line x1="{px(0.0):.1f}" y1="{oy-ph:.1f}" x2="{px(0.0):.1f}" y2="{oy:.1f}" '
                 f'stroke="#8b949e" stroke-width="1" stroke-dasharray="2 3"/>')
    parts.append(f'<text x="{ox+pw/2:.0f}" y="{oy+28}" fill="#8b949e" font-size="12" '
                 f'text-anchor="middle">difference in means under the null</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
