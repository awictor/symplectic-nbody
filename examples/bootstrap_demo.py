"""Demo: the bootstrap -- confidence intervals by resampling, for any statistic.

Builds bootstrap confidence intervals (percentile and BCa) for the mean and the median of a sample,
compares the bootstrap standard error to the analytic one, and draws the histogram of bootstrap
replicates with the interval marked.

    python examples/bootstrap_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bootstrap import (bootstrap_replicates, percentile_ci, bca_ci, bootstrap_se,  # noqa: E402
                       jackknife, mean, median, std)


class LCG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def uniform(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return (self.s >> 8) / (1 << 24)

    def normal(self, mu=0.0, sigma=1.0):
        return mu + sigma * (sum(self.uniform() for _ in range(12)) - 6.0)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    rng = LCG(42)
    data = [rng.normal(50, 8) for _ in range(80)]   # true mean 50

    print("Bootstrap: confidence intervals by resampling, no formula needed\n")
    print(f"  sample of {len(data)} draws (true mean 50): observed mean {mean(data):.3f}, "
          f"median {median(data):.3f}\n")

    print("  the MEAN (which does have a formula, so we can check ourselves):")
    bse = bootstrap_se(data, mean)
    analytic = std(data) / math.sqrt(len(data))
    print(f"    bootstrap SE {bse:.4f}  vs analytic s/sqrt(n) {analytic:.4f}")
    lo, hi = percentile_ci(data, mean)
    print(f"    95% percentile CI: [{lo:.3f}, {hi:.3f}]  (brackets 50: {lo <= 50 <= hi})")
    lo, hi = bca_ci(data, mean)
    print(f"    95% BCa CI:        [{lo:.3f}, {hi:.3f}]")
    _, bias, se = jackknife(data, mean)
    print(f"    jackknife bias {bias:.2e} (exactly 0 for the mean), SE {se:.4f}")

    print("\n  the MEDIAN (no simple closed-form CI -- the bootstrap just works):")
    lo, hi = percentile_ci(data, median)
    print(f"    95% percentile CI: [{lo:.3f}, {hi:.3f}]")
    lo, hi = bca_ci(data, median)
    print(f"    95% BCa CI:        [{lo:.3f}, {hi:.3f}]")

    print("\n  the SPREAD (standard deviation), likewise:")
    lo, hi = bca_ci(data, std)
    print(f"    95% BCa CI for the std: [{lo:.3f}, {hi:.3f}]  (true 8)")

    print("\n  The bootstrap treats the sample as a stand-in for the population: resample it with")
    print("  replacement thousands of times, recompute the statistic each time, and the spread of")
    print("  those replicates is the sampling distribution -- read off the CI from its percentiles.")
    print("  BCa additionally corrects for median bias and skewness (via the jackknife) for sharper")
    print("  coverage. It works for ANY statistic, however awkward its variance formula would be.")

    _svg(os.path.join(outdir, "bootstrap.svg"), data)
    print(f"\n  wrote {os.path.join(outdir, 'bootstrap.svg')}")


def _svg(path, data, width=760, height=380):
    reps = bootstrap_replicates(data, mean, 3000, seed=7)
    lo, hi = percentile_ci(data, mean)
    obs = mean(data)

    rmin, rmax = min(reps), max(reps)
    span = rmax - rmin if rmax > rmin else 1.0
    bins = 40
    counts = [0] * bins
    for r in reps:
        b = min(bins - 1, int((r - rmin) / span * bins))
        counts[b] += 1
    cmax = max(counts)

    ox, oy = 50, 320
    pw, ph = width - 90, 250

    def px(v):
        return ox + (v - rmin) / span * pw

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="17">'
        f'Bootstrap distribution of the mean (3000 resamples)</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'the histogram approximates the sampling distribution; the green band is the 95% CI</text>',
    ]
    # 95% CI band
    parts.append(f'<rect x="{px(lo):.1f}" y="{oy-ph:.1f}" width="{px(hi)-px(lo):.1f}" '
                 f'height="{ph}" fill="#06d6a0" opacity="0.15"/>')
    # histogram bars
    bw = pw / bins
    for b in range(bins):
        h = ph * counts[b] / cmax
        x = ox + b * bw
        parts.append(f'<rect x="{x:.1f}" y="{oy-h:.1f}" width="{bw-1:.1f}" height="{h:.1f}" '
                     f'fill="#4dabf7"/>')
    # observed estimate line
    parts.append(f'<line x1="{px(obs):.1f}" y1="{oy-ph:.1f}" x2="{px(obs):.1f}" y2="{oy:.1f}" '
                 f'stroke="#ffd43b" stroke-width="2"/>')
    parts.append(f'<text x="{px(obs):.0f}" y="{oy-ph-4:.0f}" fill="#ffd43b" font-size="11" '
                 f'text-anchor="middle">estimate</text>')
    # CI endpoints
    for v, lab in ((lo, "2.5%"), (hi, "97.5%")):
        parts.append(f'<line x1="{px(v):.1f}" y1="{oy-ph:.1f}" x2="{px(v):.1f}" y2="{oy:.1f}" '
                     f'stroke="#06d6a0" stroke-width="1.5" stroke-dasharray="3 3"/>')
    parts.append(f'<text x="{ox+pw/2:.0f}" y="{oy+28}" fill="#8b949e" font-size="12" '
                 f'text-anchor="middle">bootstrap mean</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
