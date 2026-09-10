"""Demo: the Hill sphere across the solar system.

Prints the Hill radius and stable-moon limit for each planet (in planetary radii and
compared with each planet's outermost moon), then draws the Hill radius vs orbital
distance so the balance of planet mass and solar tide is visible.

    python examples/hill_sphere_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hill_sphere import (hill_radius, stable_moon_limit,  # noqa: E402
                         AU, M_SUN, M_EARTH)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Hill sphere: r_H = a (m/3M)^(1/3) -- how far a planet holds its moons\n")
    print(f"  {'planet':>10}{'a (AU)':>9}{'mass (Me)':>11}{'r_H (Mkm)':>12}"
          f"{'stable limit':>14}")
    print("  " + "-" * 56)
    # (name, a in AU, mass in Earth masses)
    planets = [
        ("Mercury", 0.387, 0.055),
        ("Venus", 0.723, 0.815),
        ("Earth", 1.000, 1.000),
        ("Mars", 1.524, 0.107),
        ("Jupiter", 5.203, 317.8),
        ("Saturn", 9.537, 95.2),
        ("Neptune", 30.07, 17.1),
    ]
    for name, a_au, m_e in planets:
        rH = hill_radius(a_au * AU, m_e * M_EARTH, M_SUN)
        lim = stable_moon_limit(a_au * AU, m_e * M_EARTH, M_SUN)
        print(f"  {name:>10}{a_au:>9.3f}{m_e:>11.3f}{rH/1e9:>12.2f}"
              f"{lim/1e9:>12.2f}M")

    print("\n  A bigger orbit or heavier planet widens the Hill sphere (r_H ~ a m^1/3),")
    print("  so Jupiter commands a ~53-million-km domain while a hot Jupiter tucked")
    print("  against its star could barely hold a moon. Real moons survive out to only")
    print("  ~1/2 r_H prograde -- the Moon at 0.384 Mkm sits well inside Earth's 0.75")
    print("  Mkm limit, and the same math sets the mutual spacing of planetary orbits.")

    _svg(os.path.join(outdir, "hill_sphere.svg"), planets)
    print(f"\n  wrote {os.path.join(outdir, 'hill_sphere.svg')}")


def _svg(path, planets, size=720, pad=72):
    a_grid = [0.3 * (1.08 ** i) for i in range(0, 70)]   # 0.3 .. ~50 AU
    a_grid = [a for a in a_grid if a <= 40.0]
    # reference: Earth-mass planet Hill radius vs distance
    rH = [hill_radius(a * AU, M_EARTH, M_SUN) / 1e9 for a in a_grid]  # Mkm
    lx = [math.log10(a) for a in a_grid]
    ly = [math.log10(r) for r in rH]
    xmin, xmax = lx[0], lx[-1]
    # include the real planets' points in the y-range
    pts = [(math.log10(a), math.log10(hill_radius(a * AU, m * M_EARTH, M_SUN) / 1e9))
           for _, a, m in planets]
    ymin = min(min(ly), min(p[1] for p in pts))
    ymax = max(max(ly), max(p[1] for p in pts))

    def sx(x):
        return pad + (x - xmin) / (xmax - xmin) * (size - 2 * pad)

    def sy(y):
        return size - pad - (y - ymin) / (ymax - ymin) * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    poly = " ".join(f"{sx(lx[i]):.1f},{sy(ly[i]):.1f}" for i in range(len(a_grid)))
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#30363d" '
                 f'stroke-width="1.6" stroke-dasharray="4 4"/>')
    parts.append(f'<text x="{sx(lx[-6]):.1f}" y="{sy(ly[-6])-6:.1f}" fill="#8b949e" '
                 f'font-size="10">Earth-mass reference (r_H ~ a)</text>')

    for name, a, m in planets:
        px = sx(math.log10(a))
        py = sy(math.log10(hill_radius(a * AU, m * M_EARTH, M_SUN) / 1e9))
        col = "#ffd43b" if m > 50 else ("#4dabf7" if m > 0.5 else "#ff6b6b")
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{col}"/>')
        parts.append(f'<text x="{px+7:.1f}" y="{py+4:.1f}" fill="{col}" '
                     f'font-size="10">{name}</text>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Hill radius across the solar system</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'r_H ~ a m^(1/3): the giants (yellow) command the widest domains</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+24}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">log10 orbital distance (AU) -&gt;</text>')
    parts.append(f'<text x="{pad-18}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'log10 Hill radius (million km)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
