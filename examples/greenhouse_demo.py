"""Demo: the greenhouse effect across the terrestrial planets.

Prints equilibrium and surface temperatures and the greenhouse warming for Venus,
Earth and Mars, then draws surface temperature vs infrared optical depth, marking
each planet on the T_surf = T_eq (1 + 3 tau/4)^(1/4) curve.

    python examples/greenhouse_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from greenhouse import (equilibrium_temperature, surface_temperature,  # noqa: E402
                        greenhouse_warming, optical_depth_for_surface)

# (name, L_lsun, d_au, albedo, observed surface T)
PLANETS = [
    ("Venus", 1.0, 0.723, 0.77, 737.0),
    ("Earth", 1.0, 1.000, 0.30, 288.0),
    ("Mars", 1.0, 1.524, 0.25, 210.0),
    ("Titan", 1.0, 9.58, 0.22, 94.0),
]


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Greenhouse effect: T_surf = T_eq (1 + 3 tau / 4)^(1/4)\n")
    print(f"  {'planet':>8}{'T_eq (K)':>10}{'T_surf (K)':>12}"
          f"{'warming':>10}{'tau needed':>12}")
    print("  " + "-" * 52)
    for name, L, d, A, Tsurf in PLANETS:
        Te = equilibrium_temperature(L, d, A)
        tau = optical_depth_for_surface(Te, Tsurf)
        warm = Tsurf - Te
        print(f"  {name:>8}{Te:>10.1f}{Tsurf:>12.1f}{warm:>10.1f}{tau:>12.1f}")

    print("\n  Earth's modest tau ~ 0.8 lifts its 255 K skin temperature to a")
    print("  life-friendly 288 K -- a 33 K blanket. Venus, wrapped in a dense CO2")
    print("  atmosphere of tau ~ 150, is heated from a 227 K equilibrium to a")
    print("  lead-melting 737 K: the runaway greenhouse. Nearly airless Mars and")
    print("  hazy-but-cold Titan sit close to their equilibrium temperatures.")

    _svg(os.path.join(outdir, "greenhouse.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'greenhouse.svg')}")


def _svg(path, size=720, pad=70):
    # T_surf / T_eq vs tau (log tau), a single universal curve
    taus = [10 ** (-2 + 0.05 * i) for i in range(0, 90)]   # 0.01 .. ~2500
    ratio = [(1.0 + 0.75 * t) ** 0.25 for t in taus]
    lx = [math.log10(t) for t in taus]
    ly = [math.log10(r) for r in ratio]
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
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    poly = " ".join(f"{sx(lx[i]):.1f},{sy(ly[i]):.1f}" for i in range(len(taus)))
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#06d6a0" stroke-width="2.6"/>')

    cols = {"Venus": "#ffd43b", "Earth": "#4dabf7", "Mars": "#ff6b6b", "Titan": "#b197fc"}
    for name, L, d, A, Tsurf in PLANETS:
        Te = equilibrium_temperature(L, d, A)
        tau = optical_depth_for_surface(Te, Tsurf)
        if tau < taus[0]:
            tau = taus[0]
        r = (1.0 + 0.75 * tau) ** 0.25
        px = sx(math.log10(tau))
        py = sy(math.log10(r))
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{cols[name]}"/>')
        parts.append(f'<text x="{px+8:.1f}" y="{py+4:.1f}" fill="{cols[name]}" '
                     f'font-size="11">{name} (tau={tau:.1f})</text>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Greenhouse warming vs infrared optical depth</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'T_surf / T_eq = (1 + 3 tau/4)^(1/4): Earth mild, Venus runaway</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+24}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">log10 IR optical depth tau -&gt;</text>')
    parts.append(f'<text x="{pad-16}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'log10 (T_surf / T_eq)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
