"""Demo: Gibbs sampling a correlated bivariate Gaussian one coordinate at a time.

Samples a correlated 2-D Gaussian by resampling each coordinate from its conditional, shows the
recovered mean/covariance/correlation match the target, and draws the sample cloud with the target
covariance ellipse.

    python examples/gibbs_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gibbs import bivariate_gaussian, multivariate_gaussian, sample_mean, sample_cov  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Gibbs sampling: draw a joint distribution via its conditionals\n")

    mean = [2.0, -1.0]
    cov = [[3.0, 1.8], [1.8, 2.0]]
    samples = bivariate_gaussian(mean, cov, 30000, burn_in=1000, seed=3)
    m = sample_mean(samples)
    c = sample_cov(samples)
    corr = c[0][1] / math.sqrt(c[0][0] * c[1][1])
    true_corr = cov[0][1] / math.sqrt(cov[0][0] * cov[1][1])

    print(f"  Correlated bivariate Gaussian (target mean {mean}, cov {cov}):")
    print(f"    sampled mean:       [{m[0]:.3f}, {m[1]:.3f}]")
    print(f"    sampled covariance: [[{c[0][0]:.2f}, {c[0][1]:.2f}], [{c[1][0]:.2f}, {c[1][1]:.2f}]]")
    print(f"    correlation: {corr:.3f} (target {true_corr:.3f})")
    print(f"    Each step resamples x | y then y | x from their 1-D Gaussian conditionals -- every")
    print(f"    proposal accepted (acceptance ratio exactly 1), so no step is wasted.")

    # a 4-D Gaussian
    mean4 = [0, 1, 2, 3]
    cov4 = [[2.0, 0.6, 0.3, 0.1], [0.6, 1.5, 0.4, 0.2],
            [0.3, 0.4, 1.0, 0.5], [0.1, 0.2, 0.5, 1.2]]
    s4 = multivariate_gaussian(mean4, cov4, 40000, burn_in=3000, seed=5)
    m4 = sample_mean(s4)
    c4 = sample_cov(s4)
    print(f"\n  4-D Gaussian: sampled mean [{', '.join(f'{v:.2f}' for v in m4)}] (target {mean4})")
    print(f"    diagonal variances: [{', '.join(f'{c4[i][i]:.2f}' for i in range(4))}] "
          f"(target {[cov4[i][i] for i in range(4)]})")

    print("\n  Gibbs sampling is the engine of Bayesian computation: even when the joint is")
    print("  intractable, the one-dimensional conditionals are usually simple, and cycling through")
    print("  them builds a Markov chain that converges to the joint after burn-in.")

    _svg(os.path.join(outdir, "gibbs.svg"), samples, mean, cov)
    print(f"\n  wrote {os.path.join(outdir, 'gibbs.svg')}")


def _svg(path, samples, mean, cov, width=760, height=430):
    m_left, m_bot, m_top, m_right = 60, 55, 80, 40
    pw = width - m_left - m_right
    ph = height - m_top - m_bot

    xs = [s[0] for s in samples]
    ys = [s[1] for s in samples]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    def px(x):
        return m_left + (x - xmin) / (xmax - xmin) * pw

    def py(y):
        return m_top + ph - (y - ymin) / (ymax - ymin) * ph

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">'
        f'Gibbs samples of a correlated bivariate Gaussian</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'blue dots = samples; yellow = target covariance ellipse (2 sigma); the cloud tilts with the correlation</text>',
    ]

    # subsample dots
    step = max(1, len(samples) // 3000)
    for s in samples[::step]:
        parts.append(f'<circle cx="{px(s[0]):.1f}" cy="{py(s[1]):.1f}" r="1.2" '
                     f'fill="#4dabf7" fill-opacity="0.4"/>')

    # target covariance ellipse (2-sigma) via eigen-decomposition of cov
    a, b, d = cov[0][0], cov[0][1], cov[1][1]
    tr = a + d
    det = a * d - b * b
    l1 = tr / 2 + math.sqrt(max(tr * tr / 4 - det, 0))
    l2 = tr / 2 - math.sqrt(max(tr * tr / 4 - det, 0))
    # eigenvector angle
    theta = 0.5 * math.atan2(2 * b, a - d)
    pts = []
    for k in range(65):
        t = 2 * math.pi * k / 64
        ex = 2 * math.sqrt(l1) * math.cos(t)
        ey = 2 * math.sqrt(l2) * math.sin(t)
        gx = mean[0] + ex * math.cos(theta) - ey * math.sin(theta)
        gy = mean[1] + ex * math.sin(theta) + ey * math.cos(theta)
        pts.append(f"{px(gx):.1f},{py(gy):.1f}")
    parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#ffd43b" stroke-width="2"/>')
    parts.append(f'<circle cx="{px(mean[0]):.1f}" cy="{py(mean[1]):.1f}" r="4" fill="#ff6b6b"/>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
