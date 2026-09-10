"""Demo: neutron-star structure and the maximum mass (TOV).

Integrates a polytropic neutron star with the general-relativistic TOV equation
across central densities, traces the mass-radius relation (which turns over at a
maximum mass), and contrasts it with the Newtonian version, which has no maximum.
Renders the mass-radius curve to SVG.

    python examples/tov_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tov import mass_radius_sequence, maximum_mass  # noqa: E402

K, GAMMA = 100.0, 2.0


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    rcs = [10 ** (-3 + 0.12 * i) for i in range(26)]
    tov = mass_radius_sequence(rcs, K, GAMMA)
    newt = mass_radius_sequence(rcs, K, GAMMA, newtonian=True)

    m_max = maximum_mass(tov)
    print("Neutron-star structure: the TOV maximum mass\n")
    print(f"  {'rho_c':>10}{'R (km)':>10}{'M_TOV':>10}{'M_Newton':>12}")
    print("  " + "-" * 42)
    for i in range(0, len(rcs), 3):
        print(f"  {rcs[i]:>10.1e}{tov[i][0]:>10.2f}{tov[i][1]:>10.3f}{newt[i][1]:>12.2f}")
    print(f"\n  TOV maximum mass       : {m_max:.3f} M_sun (the sequence turns over)")
    print(f"  Newtonian, densest star: {newt[-1][1]:.0f} M_sun (no limit -- grows forever)")
    print("\n  General relativity imposes a maximum neutron-star mass. Above it the")
    print("  star cannot support itself and collapses to a black hole.")

    _svg(tov, newt, m_max, os.path.join(outdir, "tov.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'tov.svg')}")


def _svg(tov, newt, m_max, path, size=720, pad=64):
    # plot only the physically sensible radius window
    tov_f = [(r, m) for r, m in tov if r < 20.0]
    rmax = 16.0
    mmax = m_max * 1.25

    def sx(r):
        return pad + min(r, rmax) / rmax * (size - 2 * pad)

    def sy(m):
        return size - pad - min(m, mmax) / mmax * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{sy(m_max):.1f}" x2="{size-pad}" y2="{sy(m_max):.1f}" '
        f'stroke="#e63946" stroke-dasharray="6,4" stroke-width="1.2"/>',
        f'<text x="{size-pad-4}" y="{sy(m_max)-6:.1f}" fill="#e63946" font-size="12" '
        f'text-anchor="end">TOV maximum mass {m_max:.2f} M_sun</text>',
    ]
    poly = " ".join(f"{sx(r):.1f},{sy(m):.1f}" for r, m in tov_f)
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#4cc9f0" stroke-width="2"/>')
    for r, m in tov_f:
        parts.append(f'<circle cx="{sx(r):.1f}" cy="{sy(m):.1f}" r="2.5" fill="#4cc9f0"/>')
    parts.append(f'<text x="{pad}" y="36" fill="#e6edf3" font-size="18">'
                 f'Neutron-star mass-radius relation (TOV)</text>')
    parts.append(f'<text x="{pad}" y="{size-pad+24}" fill="#8b949e" font-size="11">'
                 f'radius (km); the curve turns over at the maximum mass</text>')
    parts.append(f'<text x="{pad-10}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'mass (M_sun)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
