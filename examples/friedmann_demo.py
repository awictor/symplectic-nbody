"""Demo: the expansion history of the universe (Friedmann cosmology).

Integrates the scale factor a(t) for matter-, radiation-, and dark-energy-
dominated universes plus a realistic flat LCDM model, prints the age of the
universe and the expansion exponent in each era, and renders a(t) curves to SVG.

    python examples/friedmann_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from friedmann import Cosmology, local_slope  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Friedmann cosmology: the expansion of the universe (time in 1/H0)\n")

    models = [
        ("radiation", Cosmology(Omega_r=1, Omega_m=0, Omega_L=0), "#f4a261"),
        ("matter", Cosmology(Omega_r=0, Omega_m=1, Omega_L=0), "#e63946"),
        ("dark energy", Cosmology(Omega_r=0, Omega_m=0, Omega_L=1), "#8338ec"),
        ("flat LCDM", Cosmology(Omega_r=0, Omega_m=0.3, Omega_L=0.7), "#4cc9f0"),
    ]

    lcdm = models[-1][1]
    print(f"  age of a flat LCDM universe : {lcdm.age():.3f}/H0")
    print(f"    (with H0 = 70 km/s/Mpc this is {lcdm.age()/70*978:.1f} Gyr)\n")

    print(f"  {'universe':<14}{'expansion exponent a ~ t^n':>28}")
    print("  " + "-" * 42)
    curves = []
    for name, c, col in models:
        a0 = 1.0 if name == "dark energy" else 1e-3
        ts, a = c.integrate_forward(a0=a0, t_max=2.5, dt=1e-4)
        curves.append((name, ts, a, col))
        if name == "dark energy":
            note = "exponential (accelerating)"
        else:
            note = f"n ~ {local_slope(ts, a, len(ts)//2):.2f}"
        print(f"  {name:<14}{note:>28}")

    print("\n  Radiation gives t^1/2, matter t^2/3, dark energy exponential growth.")
    print("  Our universe (LCDM) coasted through matter domination and is now")
    print("  entering the accelerating dark-energy era.")

    _svg(curves, os.path.join(outdir, "friedmann.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'friedmann.svg')}")


def _svg(curves, path, size=720, pad=56, t_max=2.5, a_max=6.0):
    def sx(t):
        return pad + t / t_max * (size - 2 * pad)

    def sy(a):
        return size - pad - min(a, a_max) / a_max * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{sy(1):.1f}" x2="{size-pad}" y2="{sy(1):.1f}" '
        f'stroke="#30363d" stroke-dasharray="3,4"/>',
        f'<text x="{size-pad-4}" y="{sy(1)-6:.1f}" fill="#8b949e" font-size="10" '
        f'text-anchor="end">a=1 (today)</text>',
    ]
    y = pad + 8
    for name, ts, a, col in curves:
        poly = " ".join(f"{sx(ts[i]):.1f},{sy(a[i]):.1f}"
                        for i in range(len(ts)) if ts[i] <= t_max)
        parts.append(f'<polyline points="{poly}" fill="none" stroke="{col}" stroke-width="1.8"/>')
        parts.append(f'<text x="{pad+14}" y="{y}" fill="{col}" font-size="12">{name}</text>')
        y += 16
    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Expansion of the universe: scale factor a(t)</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+22}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">time (1/H0) -&gt;</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
