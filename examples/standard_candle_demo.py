"""Demo: standard candles and the cosmic distance ladder.

Prints the distance modulus of landmark objects and the Cepheid period-luminosity
relation, then draws the modulus-vs-distance line with the rungs of the ladder --
parallax, Cepheids, Type Ia supernovae -- marked across the reachable range.

    python examples/standard_candle_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from standard_candle import (distance_modulus, distance_from_modulus,  # noqa: E402
                             cepheid_absolute_magnitude, flux_ratio)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Standard candles: m - M = 5 log10(d/10 pc); brightness gives distance\n")
    print(f"  {'object':>22}{'distance':>14}{'modulus':>10}")
    print("  " + "-" * 46)
    objs = [
        ("10 pc (M = m)", 10.0),
        ("Hyades cluster", 47.0),
        ("Galactic centre", 8200.0),
        ("LMC", 50000.0),
        ("Andromeda (M31)", 7.78e5),
        ("Virgo cluster", 1.65e7),
        ("SN Ia horizon", 1e9),
    ]
    for name, d in objs:
        dstr = f"{d:.0f} pc" if d < 1e4 else (f"{d/1e3:.0f} kpc" if d < 1e6 else f"{d/1e6:.1f} Mpc")
        print(f"  {name:>22}{dstr:>14}{distance_modulus(d):>10.2f}")

    print("\n  Cepheid period-luminosity (Leavitt's law): longer period = brighter")
    print(f"  {'period (d)':>14}{'M_V':>10}")
    for P in (3, 10, 30, 100):
        print(f"  {P:>14}{cepheid_absolute_magnitude(P):>10.2f}")

    print("\n  Five magnitudes is exactly 100x in flux. Knowing M -- from a Cepheid's")
    print("  pulsation period or a Type Ia's standardizable peak (M ~ -19.3) -- turns")
    print("  the apparent brightness into a distance. Chaining parallax to Cepheids to")
    print("  supernovae is the ladder that measures the expanding universe.")

    _svg(os.path.join(outdir, "standard_candle.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'standard_candle.svg')}")


def _svg(path, size=720, pad=70):
    ds = [10 ** (1 + 0.1 * i) for i in range(0, 91)]   # 10 pc .. 1e10 pc
    mus = [distance_modulus(d) for d in ds]
    lx = [math.log10(d) for d in ds]
    xmin, xmax = lx[0], lx[-1]
    ymin, ymax = min(mus), max(mus)

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
    # ladder-rung shaded ranges (in pc)
    rungs = [("parallax", 1e1, 1e3, "#4dabf7"),
             ("Cepheids", 1e3, 3e7, "#ffd43b"),
             ("Type Ia SNe", 3e6, 1e10, "#ff6b6b")]
    for label, d0, d1, col in rungs:
        x0, x1 = sx(math.log10(d0)), sx(math.log10(min(d1, ds[-1])))
        parts.append(f'<rect x="{x0:.1f}" y="{pad}" width="{x1-x0:.1f}" '
                     f'height="{size-2*pad:.1f}" fill="{col}" fill-opacity="0.07"/>')

    poly = " ".join(f"{sx(lx[i]):.1f},{sy(mus[i]):.1f}" for i in range(len(ds)))
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#8338ec" stroke-width="2.6"/>')

    marks = [("LMC", 5e4, "#06d6a0"), ("M31", 7.78e5, "#4dabf7"),
             ("Virgo", 1.65e7, "#ff922b")]
    for name, d, col in marks:
        px = sx(math.log10(d))
        py = sy(distance_modulus(d))
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{col}"/>')
        parts.append(f'<text x="{px+8:.1f}" y="{py+4:.1f}" fill="{col}" '
                     f'font-size="11">{name}</text>')

    ytop = pad + 22
    for i, (label, d0, d1, col) in enumerate(rungs):
        parts.append(f'<text x="{pad+10}" y="{ytop + i*16}" fill="{col}" '
                     f'font-size="12">{label}</text>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'The cosmic distance ladder: modulus vs distance</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+24}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">log10 distance (pc) -&gt;</text>')
    parts.append(f'<text x="{pad-16}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'distance modulus m - M</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
