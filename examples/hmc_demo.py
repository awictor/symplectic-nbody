"""Demo: Hamiltonian Monte Carlo vs random-walk Metropolis on a correlated Gaussian.

Samples a strongly-correlated 2-D Gaussian with both HMC and random-walk Metropolis, compares how
well each recovers the target and how fast it mixes (autocorrelation), and draws the two sample
clouds.

    python examples/hmc_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hmc import (  # noqa: E402
    hmc,
    random_walk_metropolis,
    sample_mean,
    sample_cov,
    autocorrelation,
)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Hamiltonian Monte Carlo: gradient-guided sampling of a correlated Gaussian\n")

    rho = 0.9
    det = 1 - rho * rho
    P = [[1 / det, -rho / det], [-rho / det, 1 / det]]

    def logp(q):
        x, y = q
        return -0.5 * (P[0][0] * x * x + 2 * P[0][1] * x * y + P[1][1] * y * y)

    def grad(q):
        x, y = q
        return [-(P[0][0] * x + P[0][1] * y), -(P[1][0] * x + P[1][1] * y)]

    n = 3000
    print(f"  Target: 2-D Gaussian with correlation {rho} (a narrow diagonal ridge).\n")

    hmc_s, hmc_rate = hmc(logp, [0.0, 0.0], n, step_size=0.2, n_leapfrog=25, grad=grad,
                          seed=7, burn_in=500)
    rw_s, rw_rate = random_walk_metropolis(logp, [0.0, 0.0], n, step_size=0.3, seed=7, burn_in=500)

    def report(name, s, rate):
        cov = sample_cov(s)
        r = cov[0][1] / math.sqrt(cov[0][0] * cov[1][1])
        ac = abs(autocorrelation(s, 0, 5))
        print(f"  {name:>20}: accept {rate:.2f}  recovered rho {r:+.3f}  "
              f"lag-5 autocorr {ac:.3f}")

    print(f"  {'':>20}  {'':>10}  true rho {rho}")
    report("HMC", hmc_s, hmc_rate)
    report("random-walk Metropolis", rw_s, rw_rate)

    print("\n  Both are correct in the limit, but HMC's momentum carries it along the ridge in long")
    print("  informed sweeps, so its samples decorrelate far faster -- lower autocorrelation means")
    print("  more effective samples per step. Random walk shuffles slowly across the narrow ridge.")

    _svg(os.path.join(outdir, "hmc.svg"), hmc_s, rw_s)
    print(f"\n  wrote {os.path.join(outdir, 'hmc.svg')}")


def _svg(path, hmc_s, rw_s, width=760, height=390):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        '<text x="20" y="26" fill="#e6edf3" font-size="15">'
        'Samples from a correlated Gaussian: HMC (left) and random-walk Metropolis (right)</text>',
    ]
    lim = 4.0

    def panel(samples, x0, title, col):
        size = 300
        y0 = 50
        parts.append(f'<rect x="{x0}" y="{y0}" width="{size}" height="{size}" fill="#161b22" '
                     f'stroke="#30363d"/>')

        def sx(v):
            return x0 + size * (v + lim) / (2 * lim)

        def sy(v):
            return y0 + size * (1 - (v + lim) / (2 * lim))
        # subsample for the picture
        step = max(1, len(samples) // 800)
        for i in range(0, len(samples), step):
            x, y = samples[i]
            if -lim <= x <= lim and -lim <= y <= lim:
                parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="1.6" fill="{col}" '
                             f'opacity="0.5"/>')
        parts.append(f'<text x="{x0+size/2:.0f}" y="{y0+size+22:.0f}" fill="{col}" '
                     f'font-size="11" text-anchor="middle">{title}</text>')

    panel(hmc_s, 60, "HMC (fast mixing)", "#06d6a0")
    panel(rw_s, 420, "random-walk Metropolis", "#ff922b")
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
