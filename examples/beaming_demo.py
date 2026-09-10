"""Demo: relativistic beaming of jets.

Prints the Doppler factor, flux boost and jet/counter-jet ratio for jets of various
Lorentz factors and viewing angles, then draws the Doppler factor vs viewing angle,
showing the sharp forward beaming cone that brightens the approaching jet.

    python examples/beaming_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from beaming import (lorentz_factor, doppler_factor, flux_boost,  # noqa: E402
                     beaming_cone_halfangle, jet_counterjet_ratio,
                     apparent_transverse_speed)


def _beta(gamma):
    return math.sqrt(1.0 - 1.0 / gamma ** 2)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Relativistic beaming: D = 1/(gamma(1 - beta cos theta)); flux ~ D^(3+a)\n")
    print(f"  {'gamma':>7}{'theta':>8}{'Doppler D':>11}{'flux boost':>12}"
          f"{'jet/cj':>12}{'v_app/c':>10}")
    print("  " + "-" * 60)
    for gamma in (2.0, 5.0, 10.0):
        b = _beta(gamma)
        for th_deg in (5.0, 20.0):
            th = math.radians(th_deg)
            D = doppler_factor(b, th)
            fb = flux_boost(b, th)
            R = jet_counterjet_ratio(b, th)
            va = apparent_transverse_speed(b, th)
            print(f"  {gamma:>7.0f}{th_deg:>7.0f}d{D:>11.2f}{fb:>12.3g}"
                  f"{R:>12.2g}{va:>10.1f}")

    print("\n  Aberration sweeps the emission into a cone of half-angle ~1/gamma, and the")
    print("  Doppler shift boosts the flux by D^(3+alpha). An approaching jet is brightened")
    print("  hundreds of times while its receding twin is dimmed by the same powers, which")
    print("  is why M87's jet looks one-sided. The same geometry makes blobs appear to move")
    print("  faster than light -- superluminal motion, an illusion of light-travel time.")

    _svg(os.path.join(outdir, "beaming.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'beaming.svg')}")


def _svg(path, size=720, pad=72):
    thetas = [i for i in range(0, 181)]
    curves = [(2.0, "#4dabf7", "gamma = 2"), (5.0, "#ffd43b", "gamma = 5"),
              (10.0, "#ff6b6b", "gamma = 10")]
    data = []
    for gamma, col, label in curves:
        b = _beta(gamma)
        ys = [math.log10(doppler_factor(b, math.radians(t))) for t in thetas]
        data.append((ys, col, label, gamma))

    xmin, xmax = 0.0, 180.0
    allv = [y for ys, _, _, _ in data for y in ys]
    ymin, ymax = min(allv), max(allv)

    def sx(x):
        return pad + (x - xmin) / (xmax - xmin) * (size - 2 * pad)

    def sy(y):
        return size - pad - (y - ymin) / (ymax - ymin) * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
    ]
    # D = 1 line
    if ymin <= 0.0 <= ymax:
        y1 = sy(0.0)
        parts.append(f'<line x1="{pad}" y1="{y1:.1f}" x2="{size-pad}" y2="{y1:.1f}" '
                     f'stroke="#30363d" stroke-width="1" stroke-dasharray="3 4"/>')
        parts.append(f'<text x="{size-pad-4:.1f}" y="{y1-5:.1f}" fill="#8b949e" '
                     f'font-size="10" text-anchor="end">D = 1</text>')

    parts.append(f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>')
    parts.append(f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>')

    ytop = pad + 22
    for i, (ys, col, label, gamma) in enumerate(data):
        poly = " ".join(f"{sx(thetas[j]):.1f},{sy(ys[j]):.1f}" for j in range(len(thetas)))
        parts.append(f'<polyline points="{poly}" fill="none" stroke="{col}" stroke-width="2.4"/>')
        parts.append(f'<text x="{pad+10}" y="{ytop + i*16}" fill="{col}" '
                     f'font-size="12">{label} (cone {math.degrees(beaming_cone_halfangle(_beta(gamma))):.0f} deg)</text>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Doppler factor vs viewing angle</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'faster jets beam into a narrower, brighter forward cone</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+24}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">viewing angle theta (deg) -&gt;</text>')
    parts.append(f'<text x="{pad-18}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'log10 Doppler factor D</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
