"""Demo: the white-dwarf mass-radius relation and the Chandrasekhar limit.

Integrates white dwarfs with the full relativistic degenerate electron equation
of state across a range of central densities, showing the mass climbing toward
the Chandrasekhar limit (~1.44 M_sun) as the radius shrinks. Renders the
mass-radius curve with the limit drawn in.

    python examples/chandrasekhar_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chandrasekhar import (white_dwarf_structure, chandrasekhar_mass_solar,  # noqa: E402
                           M_SUN, OMEGA_3)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    m_ch = chandrasekhar_mass_solar(2.0)
    print("White-dwarf structure and the Chandrasekhar limit\n")
    print(f"  n=3 Lane-Emden mass factor : {OMEGA_3:.5f}")
    print(f"  Chandrasekhar mass (mu_e=2): {m_ch:.3f} M_sun (the famous 1.44)\n")

    rho_cs = [10 ** e for e in [8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13]]
    print(f"  {'rho_c (kg/m^3)':>16}{'radius (km)':>14}{'mass (M_sun)':>14}")
    print("  " + "-" * 44)
    curve = []
    for rho_c in rho_cs:
        r, m = white_dwarf_structure(rho_c, dr=2e4)
        curve.append((r / 1000.0, m / M_SUN))  # km, solar masses
        print(f"  {rho_c:>16.1e}{r/1000:>14.0f}{m/M_SUN:>14.3f}")

    print("\n  As central density rises, the electrons turn relativistic, the star")
    print("  shrinks, and its mass climbs toward -- but never past -- 1.44 M_sun.")
    print("  Beyond it there is no stable white dwarf: it collapses (type-Ia SN).")

    _svg(curve, m_ch, os.path.join(outdir, "chandrasekhar.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'chandrasekhar.svg')}")


def _svg(curve, m_ch, path, size=720, pad=64):
    rmax = max(r for r, _ in curve) * 1.05
    mmax = max(m_ch * 1.15, max(m for _, m in curve) * 1.1)

    def sx(r):
        return pad + r / rmax * (size - 2 * pad)

    def sy(m):
        return size - pad - m / mmax * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
        # Chandrasekhar limit line
        f'<line x1="{pad}" y1="{sy(m_ch):.1f}" x2="{size-pad}" y2="{sy(m_ch):.1f}" '
        f'stroke="#e63946" stroke-dasharray="6,4" stroke-width="1.2"/>',
        f'<text x="{size-pad-4}" y="{sy(m_ch)-6:.1f}" fill="#e63946" font-size="12" '
        f'text-anchor="end">Chandrasekhar limit ~1.44 M_sun</text>',
    ]
    poly = " ".join(f"{sx(r):.1f},{sy(m):.1f}" for r, m in curve)
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#4cc9f0" stroke-width="2"/>')
    for r, m in curve:
        parts.append(f'<circle cx="{sx(r):.1f}" cy="{sy(m):.1f}" r="3" fill="#4cc9f0"/>')
    parts.append(f'<text x="{pad}" y="36" fill="#e6edf3" font-size="18">'
                 f'White-dwarf mass-radius relation</text>')
    parts.append(f'<text x="{pad}" y="{size-pad+24}" fill="#8b949e" font-size="11">'
                 f'radius (km) -- smaller = denser -&gt; approaches the mass limit</text>')
    parts.append(f'<text x="{pad-10}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'mass (M_sun)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
