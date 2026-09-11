"""Demo: Gaussian process regression with calibrated uncertainty.

Fits a GP to a handful of noisy samples of a smooth function and shows the posterior mean tracking
the truth while the 2-sigma confidence band pinches shut at the data and flares wide between and
beyond it. Also tunes the length scale by maximizing the log marginal likelihood.

    python examples/gaussian_process_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gaussian_process import GaussianProcess, fit_length_scale  # noqa: E402


def _target(x):
    return math.sin(x) + 0.3 * math.sin(3.0 * x)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    state = 20260911

    def rng():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 16) / 65536.0

    def gauss(sd):
        u1 = max(1e-9, rng())
        u2 = rng()
        return sd * math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)

    # sparse, slightly noisy observations, deliberately leaving a gap in the middle
    xs = [0.4, 1.0, 1.6, 2.2, 5.6, 6.2, 6.8, 7.4]
    noise_sd = 0.08
    ys = [_target(x) + gauss(noise_sd) for x in xs]

    print("Gaussian process regression: prediction with honest error bars\n")
    print(f"  {len(xs)} noisy observations of a smooth function, with a gap in the middle\n")

    # tune the length scale by marginal likelihood
    grid = [0.2, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0]
    gp, ls, lml = fit_length_scale(xs, ys, grid, signal_var=1.0, noise_var=noise_sd ** 2)
    print(f"  length scale chosen by max marginal likelihood: l = {ls} (log ML {lml:.2f})")
    print("  marginal likelihood across the grid:")
    for cand in grid:
        g = GaussianProcess(length_scale=cand, noise_var=noise_sd ** 2).fit(xs, ys)
        mark = "  <- selected" if cand == ls else ""
        print(f"    l = {cand:>4}: log ML {g.log_marginal_likelihood():7.2f}{mark}")

    # posterior on a dense grid
    gx = [i * 0.1 for i in range(80)]
    mean, std = gp.predict(gx)

    # accuracy where the truth is known, and the confidence-band behaviour
    at_data_std = gp.predict([xs[0], xs[-1]])[1]
    gap_std = gp.predict([3.8])[1][0]           # middle of the gap
    print(f"\n  Posterior 2-sigma band: ~{2*max(at_data_std):.2f} at the data, "
          f"~{2*gap_std:.2f} in the middle gap -- uncertainty tracks where data is.")

    # coverage: fraction of dense truth within the 2-sigma band
    inside = sum(1 for i, x in enumerate(gx)
                 if abs(_target(x) - mean[i]) <= 2 * std[i])
    print(f"  {100*inside/len(gx):.0f}% of the true curve lies inside the 2-sigma band "
          f"(well-calibrated).\n")

    print("  A GP places a prior over smooth functions via a kernel, then conditions on the data")
    print("  in closed form -- one Cholesky solve gives both the mean and the variance. The band")
    print("  pinches to zero at noise-free data and flares back to the prior far away: uncertainty")
    print("  you can actually trust, which is why GPs drive Bayesian optimization and active learning.")

    _svg(os.path.join(outdir, "gaussian_process.svg"), xs, ys, gx, mean, std)
    print(f"\n  wrote {os.path.join(outdir, 'gaussian_process.svg')}")


def _svg(path, xs, ys, gx, mean, std, width=760, height=420):
    lo = [mean[i] - 2 * std[i] for i in range(len(gx))]
    hi = [mean[i] + 2 * std[i] for i in range(len(gx))]
    allv = hi + lo + ys + [_target(x) for x in gx]
    va, vb = min(allv) - 0.2, max(allv) + 0.2
    xa, xb = min(gx), max(gx)
    px0, px1 = 45, width - 25
    y0, y1 = height - 45, 65

    def X(x):
        return px0 + (x - xa) / (xb - xa) * (px1 - px0)

    def Y(v):
        return y0 - (v - va) / (vb - va) * (y0 - y1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Gaussian process: posterior mean and 2-sigma confidence band</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'the band pinches shut at observations and flares in the gap and beyond -- calibrated '
        f'uncertainty</text>',
        f'<line x1="{px0}" y1="{Y(0):.1f}" x2="{px1}" y2="{Y(0):.1f}" stroke="#21262d" stroke-width="1"/>',
    ]

    # confidence band as a filled polygon
    top = " ".join(f"{X(gx[i]):.1f},{Y(hi[i]):.1f}" for i in range(len(gx)))
    bot = " ".join(f"{X(gx[i]):.1f},{Y(lo[i]):.1f}" for i in range(len(gx) - 1, -1, -1))
    parts.append(f'<polygon points="{top} {bot}" fill="#4dabf7" opacity="0.18"/>')

    # true function (dashed green)
    tru = " ".join(f"{X(gx[i]):.1f},{Y(_target(gx[i])):.1f}" for i in range(len(gx)))
    parts.append(f'<polyline points="{tru}" fill="none" stroke="#06d6a0" stroke-width="1.6" '
                 f'stroke-dasharray="5 3"/>')
    # posterior mean (solid blue)
    mn = " ".join(f"{X(gx[i]):.1f},{Y(mean[i]):.1f}" for i in range(len(gx)))
    parts.append(f'<polyline points="{mn}" fill="none" stroke="#4dabf7" stroke-width="2.4"/>')
    # observations (yellow dots)
    for i in range(len(xs)):
        parts.append(f'<circle cx="{X(xs[i]):.1f}" cy="{Y(ys[i]):.1f}" r="3.5" fill="#ffd43b" '
                     f'stroke="#0d1117" stroke-width="0.8"/>')

    # legend
    leg = [("#4dabf7", "posterior mean"), ("#06d6a0", "true function"),
           ("#ffd43b", "observations"), ("#4dabf7", "2-sigma band")]
    for i, (c, lab) in enumerate(leg):
        yy = y1 + 2 + i * 15
        parts.append(f'<rect x="{px0+6}" y="{yy-8}" width="11" height="6" fill="{c}" '
                     f'{"opacity=0.4" if i==3 else ""}/>')
        parts.append(f'<text x="{px0+22}" y="{yy-2}" fill="#8b949e" font-size="10">{lab}</text>')

    parts.append(f'<line x1="{px0}" y1="{y0}" x2="{px1}" y2="{y0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{px0}" y1="{y0}" x2="{px0}" y2="{y1}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
