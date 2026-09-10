"""Demo: the Rossby number and geostrophic balance.

Prints the Rossby number and regime for flows from tornadoes to ocean gyres, then
draws Ro vs length scale at fixed speed, shading the geostrophic (rotation-dominated)
region below Ro = 0.1 and marking real flows.

    python examples/rossby_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rossby import (coriolis_parameter, rossby_number, is_geostrophic,  # noqa: E402
                    geostrophic_wind, deformation_radius, inertial_period)

F45 = coriolis_parameter(45.0)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Rossby number Ro = U / (f L): does planetary rotation matter?\n")
    print(f"  Coriolis parameter at 45 deg: f = {F45:.2e} s^-1")
    print(f"  inertial period there: {inertial_period(F45)/3600:.1f} hr "
          f"(half a pendulum day)\n")
    print(f"  {'flow':>20}{'U (m/s)':>9}{'L':>12}{'Ro':>12}{'regime':>16}")
    print("  " + "-" * 69)
    flows = [
        ("bathtub drain", 0.2, 0.1),
        ("tornado", 100.0, 100.0),
        ("sea breeze", 5.0, 2e4),
        ("hurricane", 50.0, 5e5),
        ("cyclone (synoptic)", 10.0, 1e6),
        ("ocean gyre", 0.1, 2e6),
    ]
    for name, U, L in flows:
        Ro = rossby_number(U, L, F45)
        regime = "geostrophic" if is_geostrophic(U, L, F45) else "ageostrophic"
        Lstr = f"{L/1e3:.0f} km" if L >= 1e3 else f"{L:.1f} m"
        print(f"  {name:>20}{U:>9.1f}{Lstr:>12}{Ro:>12.3g}{regime:>16}")

    Ug = geostrophic_wind(1e-3, 1.2, F45)
    print(f"\n  A 1 mb / 100 km pressure gradient drives a ~{Ug:.0f} m/s geostrophic wind")
    print("  -- blowing ALONG the isobars, not across them, because Coriolis balances")
    print("  the pressure force. Small, fast flows (Ro >> 1) ignore rotation; large,")
    print("  slow ones (Ro << 1) are ruled by it, which is why weather systems and")
    print("  ocean gyres are big rotating vortices rather than simple radial flows.")

    _svg(os.path.join(outdir, "rossby.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'rossby.svg')}")


def _svg(path, size=720, pad=70, U=10.0):
    Ls = [10 ** (-1 + 0.1 * i) for i in range(0, 81)]   # 0.1 m .. ~1e7 m
    Ros = [rossby_number(U, L, F45) for L in Ls]
    lx = [math.log10(L) for L in Ls]
    ly = [math.log10(R) for R in Ros]
    xmin, xmax = lx[0], lx[-1]
    ymin, ymax = min(ly), max(ly)

    def sx(x):
        return pad + (x - xmin) / (xmax - xmin) * (size - 2 * pad)

    def sy(y):
        return size - pad - (y - ymin) / (ymax - ymin) * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
    ]
    # geostrophic band Ro < 0.1 shaded
    yg = sy(math.log10(0.1))
    parts.append(f'<rect x="{pad}" y="{yg:.1f}" width="{size-2*pad}" '
                 f'height="{size-pad-yg:.1f}" fill="#4dabf7" fill-opacity="0.10"/>')
    parts.append(f'<line x1="{pad}" y1="{yg:.1f}" x2="{size-pad}" y2="{yg:.1f}" '
                 f'stroke="#4dabf7" stroke-width="1.3" stroke-dasharray="5 4"/>')
    parts.append(f'<text x="{pad+8}" y="{yg+16:.1f}" fill="#4dabf7" '
                 f'font-size="11">Ro &lt; 0.1: geostrophic (rotation rules)</text>')

    parts.append(f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>')
    parts.append(f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>')

    poly = " ".join(f"{sx(lx[i]):.1f},{sy(ly[i]):.1f}" for i in range(len(Ls)))
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#ffd43b" stroke-width="2.6"/>')

    for name, L, col in [("tornado", 100.0, "#ff6b6b"),
                         ("hurricane", 5e5, "#ff922b"),
                         ("cyclone", 1e6, "#06d6a0")]:
        Ro = rossby_number(U, L, F45)
        px = sx(math.log10(L))
        py = sy(math.log10(Ro))
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{col}"/>')
        parts.append(f'<text x="{px+8:.1f}" y="{py:.1f}" fill="{col}" '
                     f'font-size="11">{name}</text>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Rossby number vs length scale (U = 10 m/s, 45 deg)</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'big slow flows are rotation-dominated; small fast ones are not</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+24}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">log10 length scale (m) -&gt;</text>')
    parts.append(f'<text x="{pad-16}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'log10 Rossby number</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
