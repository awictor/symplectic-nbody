"""Demo: a particle filter tracking a nonlinear system.

Tracks a hidden state through a nonlinear motion model from noisy measurements, showing the
weighted particle cloud's mean following the truth below the raw sensor error, and -- the crux --
how adaptive resampling keeps the effective sample size healthy where a weight-only filter's cloud
collapses to a single particle.

    python examples/particle_filter_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from particle_filter import ParticleFilter, gaussian_likelihood, _Rng  # noqa: E402


def trans(x, rng):
    return x + 0.3 * math.sin(x) + 0.5 + rng.gauss(0.0, 0.3)


def lik(x, z):
    return gaussian_likelihood(x, z, 1.0)


def init(rng):
    return rng.gauss(0.0, 1.0)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    gen = _Rng(7)
    x = 0.0
    T = 50
    truth, meas = [], []
    for _ in range(T):
        x = x + 0.3 * math.sin(x) + 0.5 + gen.gauss(0.0, 0.3)
        truth.append(x)
        meas.append(x + gen.gauss(0.0, 1.0))

    pf = ParticleFilter(trans, lik, init, n_particles=500, seed=3)
    est = pf.run(meas)

    def err(seq):
        return sum(abs(seq[t] - truth[t]) for t in range(5, T)) / (T - 5)

    raw_err = err(meas)
    pf_err = err(est)

    print("Particle filter: sequential Monte-Carlo estimation of a nonlinear system\n")
    print(f"  {T} steps, nonlinear motion x <- x + 0.3 sin(x) + 0.5, noisy position sensor\n")
    print(f"  mean absolute error vs truth:  raw measurements {raw_err:.3f}")
    print(f"                                 particle filter  {pf_err:.3f}  "
          f"({100*(1-pf_err/raw_err):.0f}% better)\n")

    # ess health with vs without resampling
    pf_nr = ParticleFilter(trans, lik, init, n_particles=500, resample_threshold=0.0, seed=3)
    pf_nr.run(meas)
    print("  Effective sample size (of 500) -- resampling is what stops degeneracy:")
    print(f"    adaptive resampling:  min ESS {min(pf.ess_history):.0f}, "
          f"{sum(pf.resampled_history)} resamples")
    print(f"    NO resampling:        min ESS {min(pf_nr.ess_history):.1f} "
          f"(cloud collapses to ~1 particle)\n")

    print("  More particles, lower error (mean over 3 seeds):")
    for n in (20, 100, 500):
        errs = []
        for s in range(3):
            p = ParticleFilter(trans, lik, init, n_particles=n, seed=s)
            e = p.run(meas)
            errs.append(sum(abs(e[t] - truth[t]) for t in range(5, T)) / (T - 5))
        print(f"    {n:>3} particles: error {sum(errs)/len(errs):.3f}")

    print("\n  Unlike the Kalman filter, a particle filter needs no linearity or Gaussian noise --")
    print("  it just pushes a cloud of samples through the true dynamics, reweights by the")
    print("  measurement likelihood, and resamples to concentrate on where the posterior lives.")

    _svg(os.path.join(outdir, "particle_filter.svg"), truth, meas, est,
         pf.ess_history, pf_nr.ess_history, pf.n)
    print(f"\n  wrote {os.path.join(outdir, 'particle_filter.svg')}")


def _svg(path, truth, meas, est, ess_adapt, ess_none, n_particles, width=760, height=440):
    T = len(truth)
    lx0, lx1 = 45, width // 2 - 15
    y0, y1 = height - 50, 70
    allv = truth + meas + est
    va, vb = min(allv) - 0.5, max(allv) + 0.5

    def LX(t):
        return lx0 + t / (T - 1) * (lx1 - lx0)

    def LY(v):
        return y0 - (v - va) / (vb - va) * (y0 - y1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Particle filter: nonlinear tracking (left), resampling saves the cloud (right)</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'left: truth vs noisy sensor vs particle-filter estimate; right: effective sample '
        f'size with and without resampling</text>',
        f'<line x1="{lx0}" y1="{y0}" x2="{lx1}" y2="{y0}" stroke="#8b949e" stroke-width="1.1"/>',
        f'<line x1="{lx0}" y1="{y0}" x2="{lx0}" y2="{y1}" stroke="#8b949e" stroke-width="1.1"/>',
    ]
    # measurements
    for t in range(T):
        parts.append(f'<circle cx="{LX(t):.1f}" cy="{LY(meas[t]):.1f}" r="2" '
                     f'fill="#8b949e" opacity="0.55"/>')

    def poly(seq, col, w):
        pts = " ".join(f"{LX(t):.1f},{LY(seq[t]):.1f}" for t in range(T))
        return f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="{w}"/>'

    parts.append(poly(truth, "#06d6a0", 2.2))
    parts.append(poly(est, "#4dabf7", 2.0))
    for i, (c, lab) in enumerate([("#06d6a0", "truth"), ("#8b949e", "measured"),
                                  ("#4dabf7", "PF estimate")]):
        yy = y1 + 4 + i * 15
        parts.append(f'<rect x="{lx0+6}" y="{yy-8}" width="11" height="6" fill="{c}"/>')
        parts.append(f'<text x="{lx0+22}" y="{yy-2}" fill="#8b949e" font-size="10">{lab}</text>')

    # right: ESS curves
    rx0, rx1 = width // 2 + 45, width - 30
    ry0, ry1 = y0, y1

    def RX(t):
        return rx0 + t / (T - 1) * (rx1 - rx0)

    def RY(v):
        return ry0 - v / n_particles * (ry0 - ry1)

    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx1}" y2="{ry0}" stroke="#8b949e" stroke-width="1.1"/>')
    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx0}" y2="{ry1}" stroke="#8b949e" stroke-width="1.1"/>')
    # threshold line at N/2
    parts.append(f'<line x1="{rx0}" y1="{RY(n_particles/2):.1f}" x2="{rx1}" y2="{RY(n_particles/2):.1f}" '
                 f'stroke="#30363d" stroke-width="1" stroke-dasharray="3 3"/>')
    parts.append(f'<text x="{rx1-2:.1f}" y="{RY(n_particles/2)-3:.1f}" fill="#8b949e" font-size="9" '
                 f'text-anchor="end">resample threshold N/2</text>')
    ea = " ".join(f"{RX(t):.1f},{RY(ess_adapt[t]):.1f}" for t in range(T))
    parts.append(f'<polyline points="{ea}" fill="none" stroke="#06d6a0" stroke-width="2"/>')
    en = " ".join(f"{RX(t):.1f},{RY(ess_none[t]):.1f}" for t in range(T))
    parts.append(f'<polyline points="{en}" fill="none" stroke="#ff6b6b" stroke-width="2"/>')
    parts.append(f'<text x="{rx0+6}" y="{ry1+4}" fill="#06d6a0" font-size="10">with resampling</text>')
    parts.append(f'<text x="{rx0+6}" y="{ry1+18}" fill="#ff6b6b" font-size="10">without (collapses)</text>')
    parts.append(f'<text x="{(rx0+rx1)/2:.1f}" y="{ry0+18:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">effective sample size vs step</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
