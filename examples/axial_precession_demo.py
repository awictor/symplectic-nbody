"""Demo: precession of the equinoxes (Earth's axial precession).

Prints the Sun/Moon/combined precession rates and the ~25,800-year period, then
draws the circle the celestial pole traces on the sky over one Great Year, marking
Polaris (now), Thuban (~2700 BC), and Vega (~14000 AD).

    python examples/axial_precession_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from axial_precession import (torque_precession_rate, lunisolar_rate,  # noqa: E402
                              precession_period_years, rate_arcsec_per_year,
                              moon_to_sun_ratio, M_SUN, M_MOON, AU, R_MOON,
                              OBLIQUITY)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    sun = torque_precession_rate(M_SUN, AU)
    moon = torque_precession_rate(M_MOON, R_MOON)
    both = lunisolar_rate()

    print("Precession of the equinoxes (luni-solar torque on Earth's bulge)\n")
    print(f"  {'source':>10}{'arcsec/yr':>14}{'period (yr)':>16}")
    print("  " + "-" * 40)
    for name, rate in (("Sun", sun), ("Moon", moon), ("Sun+Moon", both)):
        print(f"  {name:>10}{rate_arcsec_per_year(rate):>14.2f}"
              f"{precession_period_years(rate):>16.0f}")
    print(f"\n  Moon/Sun torque ratio: {moon_to_sun_ratio():.2f} "
          f"(nearby body wins: torque ~ M / r^3)")
    print(f"  obliquity: {math.degrees(OBLIQUITY):.2f} deg")
    print(f"  measured luni-solar precession: 50.29 arcsec/yr, period ~25772 yr")
    print("\n  The pole traces a 47-deg-wide circle on the sky, so Polaris is only")
    print("  our temporary North Star -- Vega held the title ~12000 BC and will")
    print("  again ~14000 AD. The same drift slips the equinox one zodiac sign")
    print("  every ~2150 years (the astrological 'ages').")

    _svg(os.path.join(outdir, "axial_precession.svg"),
         rate_arcsec_per_year(both), precession_period_years(both))
    print(f"\n  wrote {os.path.join(outdir, 'axial_precession.svg')}")


def _svg(path, arcsec_yr, period_yr, size=560):
    cx = cy = size / 2.0
    r = size * 0.34
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
    ]
    # deterministic pseudo-stars (no RNG): fixed lattice hash
    for i in range(90):
        sx = (i * 97 + 31) % size
        sy = (i * 53 + 17) % size
        rad = 0.6 + (i % 3) * 0.5
        parts.append(f'<circle cx="{sx}" cy="{sy}" r="{rad:.1f}" fill="#30363d"/>')
    parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" '
                 f'fill="none" stroke="#8338ec" stroke-width="2" stroke-dasharray="5 4"/>')
    parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="2.5" fill="#8b949e"/>')
    parts.append(f'<text x="{cx+6:.1f}" y="{cy-6:.1f}" fill="#8b949e" '
                 f'font-size="10">ecliptic pole</text>')

    marks = [("Polaris (now)", 90, "#ffd43b"),
             ("Thuban (~2700 BC)", 200, "#4dabf7"),
             ("Vega (~14000 AD)", 330, "#ff6b6b")]
    for label, deg, col in marks:
        a = math.radians(deg)
        px = cx + r * math.cos(a)
        py = cy - r * math.sin(a)
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{col}"/>')
        anchor = "end" if math.cos(a) < 0 else "start"
        dx = -8 if math.cos(a) < 0 else 8
        parts.append(f'<text x="{px+dx:.1f}" y="{py+3:.1f}" fill="{col}" '
                     f'font-size="10" text-anchor="{anchor}">{label}</text>')

    parts.append(f'<text x="20" y="30" fill="#e6edf3" font-size="17">'
                 f'The wandering celestial pole</text>')
    parts.append(f'<text x="20" y="50" fill="#8b949e" font-size="11">'
                 f'{arcsec_yr:.1f} arcsec/yr -&gt; one circuit every {period_yr:.0f} yr</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
