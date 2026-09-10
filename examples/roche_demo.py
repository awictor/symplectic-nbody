"""Demo: the Roche limit and tidal disruption of a rubble-pile satellite.

Scans a rubble satellite across the Roche limit and prints the surviving bound
fraction (a sharp transition at d ~ d_Roche), then renders a satellite being torn
into a tidal stream as it orbits inside the limit.

    python examples/roche_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from roche import (make_rubble_satellite, roche_limit, bound_fraction,  # noqa: E402
                   surviving_bound_fraction)

_BARS = " .:-=+*#@"


def bar(frac, width=30):
    return "#" * int(frac * width) + "-" * (width - int(frac * width))


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Roche limit: tidal disruption of a rubble-pile satellite\n")
    print(f"{'d / d_Roche':>12}  surviving bound fraction")
    print("-" * 48)
    for f in (0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0):
        bf = surviving_bound_fraction(f)
        print(f"{f:>12.2f}  {bar(bf)} {bf:.2f}")
    print("\nBelow ~1 Roche the satellite is shredded; above it, it survives.")

    # render a disrupting satellite: snapshot the particle stream inside Roche
    m_primary, m_sat, sat_radius = 1.0, 1e-3, 0.05
    rho_sat = m_sat / (4.0 / 3.0 * math.pi * sat_radius ** 3)
    rho_p = m_primary / (4.0 / 3.0 * math.pi * 0.3 ** 3)
    d_roche = roche_limit(0.3, rho_p, rho_sat)
    d = 0.7 * d_roche
    sysn, ip = make_rubble_satellite(d, m_primary, m_sat=m_sat, n=120,
                                     sat_radius=sat_radius, seed=6)
    period = 2 * math.pi * math.sqrt(d ** 3 / m_primary)
    dt = period / 3000
    frames = {0: "start", 1500: "half orbit", 4500: "1.5 orbits"}
    saved = {}
    for step in range(4600):
        sysn.step("verlet", dt)
        if step in frames:
            saved[frames[step]] = [(sysn.pos[i][0], sysn.pos[i][1])
                                   for i in range(sysn.n) if i != ip]
    _stream_svg(saved, sysn.pos[ip], d_roche,
                os.path.join(outdir, "roche_disruption.svg"))
    print(f"\nwrote {os.path.join(outdir, 'roche_disruption.svg')}")
    print("The rubble pile stretches into a tidal stream -- how Saturn's rings")
    print("and Shoemaker-Levy 9's fragment chain came to be.")


def _stream_svg(frames, primary_xy, d_roche, path, size=720):
    # world extent from all points
    allx = [p[0] for pts in frames.values() for p in pts] + [primary_xy[0]]
    ally = [p[1] for pts in frames.values() for p in pts] + [primary_xy[1]]
    ext = max(max(abs(v) for v in allx), max(abs(v) for v in ally)) * 1.1

    def sx(x): return size / 2 + x / ext * (size / 2 - 30)
    def sy(y): return size / 2 - y / ext * (size / 2 - 30)

    colors = {"start": "#4cc9f0", "half orbit": "#ffbe0b", "1.5 orbits": "#ff006e"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#05070d"/>',
    ]
    # Roche circle around the primary
    parts.append(f'<circle cx="{sx(0):.1f}" cy="{sy(0):.1f}" r="{d_roche/ext*(size/2-30):.1f}" '
                 f'fill="none" stroke="#30363d" stroke-dasharray="4,4"/>')
    parts.append(f'<circle cx="{sx(0):.1f}" cy="{sy(0):.1f}" r="6" fill="#ffd166"/>')
    for label, pts in frames.items():
        col = colors.get(label, "#ffffff")
        dots = "".join(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="1.6" '
                       f'fill="{col}" fill-opacity="0.85"/>' for x, y in pts)
        parts.append(dots)
    parts.append(f'<text x="16" y="26" fill="#e6edf3" font-size="16">'
                 f'Tidal disruption inside the Roche limit</text>')
    y = 46
    for label, col in colors.items():
        parts.append(f'<text x="16" y="{y}" fill="{col}" font-size="12">{label}</text>')
        y += 16
    parts.append(f'<text x="{size-16}" y="{size-14}" fill="#8b949e" font-size="11" '
                 f'text-anchor="end">dashed = Roche limit; gold = primary</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
