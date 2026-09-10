"""Demo: the Kolmogorov cascade -- the -5/3 energy spectrum of turbulence.

Prints the Kolmogorov microscales and the width of the inertial range across flows, then
draws the energy spectrum E(k) on log-log axes: the -5/3 inertial range between the large
stirring scale and the tiny dissipation scale where viscosity turns motion into heat.

    python examples/kolmogorov_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kolmogorov import (dissipation_rate, kolmogorov_length, kolmogorov_time,  # noqa: E402
                        kolmogorov_velocity, energy_spectrum, scale_separation)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Kolmogorov cascade: big eddies -> small eddies -> heat, E(k) ~ k^(-5/3)\n")
    print(f"  {'flow':<24}{'Re':>10}{'eta':>12}{'tau_eta':>11}{'L/eta':>10}")
    # (name, u, L, nu)
    flows = [
        ("stirred coffee", 0.1, 0.05, 1e-6),
        ("room air current", 0.5, 3.0, 1.5e-5),
        ("river reach", 1.0, 10.0, 1e-6),
        ("atmosphere (km)", 10.0, 1000.0, 1.5e-5),
    ]
    for name, u, L, nu in flows:
        eps = dissipation_rate(u, L)
        Re = u * L / nu
        eta = kolmogorov_length(nu, eps)
        tau = kolmogorov_time(nu, eps)
        print(f"  {name:<24}{Re:>10.1e}{eta*1000:>10.3f}mm{tau:>10.2e}s{scale_separation(Re):>10.1e}")

    print("\n  Energy is fed in at the large scale, cascades untouched through the inertial")
    print("  range as E(k) ~ k^(-5/3), and is dissipated only at the Kolmogorov scale eta")
    print("  where the eddy Reynolds number drops to 1. The range widens as Re^(3/4), so a")
    print("  weather-scale flow spans millions of eddy sizes -- why turbulence is so costly")
    print("  to simulate (~Re^(9/4) grid points in 3-D).")

    _svg(os.path.join(outdir, "kolmogorov.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'kolmogorov.svg')}")


def _svg(path, size=720, pad=80):
    # E(k) log-log with inertial -5/3 range; mark injection (k_L) and dissipation (k_eta).
    nu = 1e-5
    u, L = 1.0, 1.0
    eps = dissipation_rate(u, L)
    eta = kolmogorov_length(nu, eps)
    k_L = 2.0 * math.pi / L            # injection wavenumber
    k_eta = 2.0 * math.pi / eta        # dissipation wavenumber

    lk0, lk1 = math.log10(k_L / 3), math.log10(k_eta * 3)
    ks = [10 ** (lk0 + (lk1 - lk0) * i / 299) for i in range(300)]

    def E(k):
        base = energy_spectrum(k, eps)
        # roll off below injection (energy-containing) and above dissipation (viscous cutoff)
        if k < k_L:
            base *= (k / k_L) ** (4.0)        # steep rise of energy-containing range (~k^4)...
            # actually low-k side: E~k^? keep schematic increasing toward k_L
        if k > k_eta:
            base *= math.exp(-(k / k_eta - 1.0))   # viscous dissipation cutoff
        return base

    Es = [E(k) for k in ks]
    ly0, ly1 = math.log10(min(e for e in Es if e > 0)), math.log10(max(Es))

    x0, x1 = pad, size - pad
    y0, y1 = size - pad, pad + 44

    def X(k):
        return x0 + (math.log10(k) - lk0) / (lk1 - lk0) * (x1 - x0)

    def Y(e):
        return y0 - (math.log10(e) - ly0) / (ly1 - ly0) * (y0 - y1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="34" fill="#e6edf3" font-size="18">The Kolmogorov energy spectrum</text>',
        f'<text x="20" y="52" fill="#8b949e" font-size="12">'
        f'E(k) ~ k^(-5/3) in the inertial range between stirring and dissipation</text>',
    ]

    # shade inertial range
    parts.append(f'<rect x="{X(k_L):.1f}" y="{y1:.1f}" width="{X(k_eta)-X(k_L):.1f}" '
                 f'height="{y0-y1:.1f}" fill="#4dabf7" opacity="0.07"/>')

    # axes
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#8b949e" stroke-width="1.5"/>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#8b949e" stroke-width="1.5"/>')

    # a pure -5/3 reference line through the inertial range
    kref0, kref1 = k_L, k_eta
    ref = [(kref0, energy_spectrum(kref0, eps)), (kref1, energy_spectrum(kref1, eps))]
    parts.append(f'<line x1="{X(ref[0][0]):.1f}" y1="{Y(ref[0][1]):.1f}" '
                 f'x2="{X(ref[1][0]):.1f}" y2="{Y(ref[1][1]):.1f}" '
                 f'stroke="#ffd43b" stroke-width="1.4" stroke-dasharray="6 4"/>')
    parts.append(f'<text x="{X(math.sqrt(k_L*k_eta)):.1f}" '
                 f'y="{Y(energy_spectrum(math.sqrt(k_L*k_eta), eps))-8:.1f}" '
                 f'fill="#ffd43b" font-size="12">slope -5/3</text>')

    # the spectrum curve
    poly = " ".join(f"{X(ks[i]):.1f},{Y(Es[i]):.1f}" for i in range(len(ks)) if Es[i] > 0)
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#4dabf7" stroke-width="2.6"/>')

    # injection and dissipation markers
    for k, col, lbl in ((k_L, "#06d6a0", "injection (stirring scale L)"),
                        (k_eta, "#ff6b6b", "dissipation (Kolmogorov eta)")):
        parts.append(f'<line x1="{X(k):.1f}" y1="{y0:.1f}" x2="{X(k):.1f}" y2="{y1:.1f}" '
                     f'stroke="{col}" stroke-width="1.4" stroke-dasharray="4 4" opacity="0.7"/>')
        parts.append(f'<text x="{X(k):.1f}" y="{y1-6:.1f}" fill="{col}" font-size="10" '
                     f'text-anchor="middle">{lbl}</text>')

    parts.append(f'<text x="{(x0+x1)/2:.1f}" y="{y0+28:.1f}" fill="#8b949e" font-size="11" '
                 f'text-anchor="middle">wavenumber k (small eddies to the right)</text>')
    parts.append(f'<text x="24" y="{(y0+y1)/2:.1f}" fill="#8b949e" font-size="11" '
                 f'transform="rotate(-90 24 {(y0+y1)/2:.1f})" text-anchor="middle">energy E(k)</text>')
    parts.append(f'<text x="{X(k_L)+8:.1f}" y="{y1+30:.1f}" fill="#8b949e" font-size="10">'
                 f'energy in -&gt; cascades -&gt; heat out</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
