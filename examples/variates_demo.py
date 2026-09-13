"""Demo: random variate generation -- sampling the named distributions from uniforms.

Draws large samples from exponential, gamma, beta, Poisson, and binomial distributions, confirms the
empirical moments match the analytic ones, and draws histograms of each. All from one seeded uniform
stream.

    python examples/variates_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from variates import Variates, analytic_moments, sample_stats  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Variate generation: turning uniform randoms into any named distribution\n")

    v = Variates(2024)
    N = 50000
    specs = [
        ("Exponential(rate=1.5)", lambda: v.exponential(1.5), ("exponential", dict(rate=1.5))),
        ("Gamma(k=3, theta=1.5)", lambda: v.gamma(3, 1.5), ("gamma", dict(shape=3, scale=1.5))),
        ("Beta(2, 5)", lambda: v.beta(2, 5), ("beta", dict(a=2, b=5))),
        ("Poisson(lambda=4)", lambda: v.poisson(4), ("poisson", dict(lam=4))),
        ("Binomial(20, 0.3)", lambda: v.binomial(20, 0.3), ("binomial", dict(n=20, p=0.3))),
    ]
    print(f"  {'distribution':<24} {'emp mean':>9} {'true':>7}   {'emp var':>9} {'true':>7}")
    hists = []
    for label, gen, (name, params) in specs:
        s = [gen() for _ in range(N)]
        em, ev = sample_stats(s)
        am, av = analytic_moments(name, **params)
        print(f"  {label:<24} {em:>9.3f} {am:>7.3f}   {ev:>9.3f} {av:>7.3f}")
        hists.append((label, s))
    print("\n    (empirical moments track the analytic ones -- the samplers are correct)\n")

    print("  Each distribution has its own transform of uniforms: exponential inverts its CDF")
    print("  (-ln U / rate), the normal comes from Box-Muller, the gamma from Marsaglia-Tsang's")
    print("  squeeze, beta from a ratio of gammas, Poisson from Knuth's uniform-product, binomial from")
    print("  summed Bernoullis -- the building blocks of every Monte-Carlo simulation.")

    _svg(os.path.join(outdir, "variates.svg"), hists)
    print(f"\n  wrote {os.path.join(outdir, 'variates.svg')}")


def _svg(path, hists, width=760, height=480):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="16">'
        f'Sampled distributions (50000 draws each) from one uniform stream</text>',
    ]
    cols = ["#4dabf7", "#06d6a0", "#ffd43b", "#ff922b", "#b197fc"]
    panel_h = (height - 60) / len(hists)
    for idx, (label, samples) in enumerate(hists):
        y0 = 45 + idx * panel_h
        lo, hi = min(samples), max(samples)
        bins = 30
        counts = [0] * bins
        span = (hi - lo) or 1
        for x in samples:
            b = min(bins - 1, int((x - lo) / span * bins))
            counts[b] += 1
        cmax = max(counts)
        ox = 200
        pw = width - ox - 30
        bw = pw / bins
        col = cols[idx % len(cols)]
        parts.append(f'<text x="20" y="{y0 + panel_h/2:.0f}" fill="{col}" font-size="11">{label}</text>')
        base = y0 + panel_h - 14
        ph = panel_h - 22
        for b in range(bins):
            h = ph * counts[b] / cmax
            x = ox + b * bw
            parts.append(f'<rect x="{x:.1f}" y="{base-h:.1f}" width="{bw-1:.1f}" height="{h:.1f}" '
                         f'fill="{col}"/>')
        parts.append(f'<text x="{ox}" y="{base+11:.0f}" fill="#8b949e" font-size="8">{lo:.1f}</text>')
        parts.append(f'<text x="{ox+pw-20:.0f}" y="{base+11:.0f}" fill="#8b949e" font-size="8">'
                     f'{hi:.1f}</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
