"""Demo: osmotic pressure -- van't Hoff's law across a membrane.

Prints the osmotic pressure of everyday solutions (seawater, blood, saline, sugar) and how
it scales with the van't Hoff factor, then draws osmotic pressure vs concentration for
glucose, NaCl and CaCl2, with seawater and blood plasma marked and the reverse-osmosis
threshold shaded.

    python examples/osmosis_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from osmosis import osmotic_pressure, ATM  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Osmotic pressure: Pi = i c R T (van't Hoff -- the ideal-gas law for solutes)\n")
    print(f"  {'solution':<34}{'conc':>12}{'i':>4}{'Pi (atm)':>12}")
    cases = [
        ("glucose, 0.30 mol/L (isotonic)", 300.0, 1.0, 310.0),
        ("blood plasma (~0.30 osmol/L)", 300.0, 1.0, 310.0),
        ("normal saline 0.9% (~0.15 M NaCl)", 150.0, 2.0, 310.0),
        ("seawater (~0.6 M NaCl-equiv)", 600.0, 2.0, 288.0),
        ("Dead Sea brine (~5 M ions)", 2500.0, 2.0, 298.0),
    ]
    for label, c, i, T in cases:
        pi = osmotic_pressure(c, T, i)
        print(f"  {label:<34}{c/1000:>9.2f} M{i:>4.0f}{pi/ATM:>12.2f}")

    print("\n  Same molarity, more particles -> more pressure: CaCl2 (i~3) pushes three times")
    print("  as hard as glucose (i=1). Seawater's ~27 atm is the wall reverse-osmosis must")
    print("  beat to squeeze fresh water back out; blood's ~7.6 atm sets isotonic IV fluids.")

    _svg(os.path.join(outdir, "osmosis.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'osmosis.svg')}")


def _svg(path, size=720, pad=78):
    T = 298.0
    c_max = 700.0            # mol/m^3 (0.70 mol/L)
    n = 120
    solutes = [
        ("glucose (i=1)", 1.0, "#4dabf7"),
        ("NaCl (i=2)", 2.0, "#06d6a0"),
        ("CaCl2 (i=3)", 3.0, "#ff922b"),
    ]
    pi_max = osmotic_pressure(c_max, T, 3.0) / ATM * 1.05

    x0, x1 = pad, size - pad
    y0, y1 = size - pad, pad + 34

    def X(c):
        return x0 + c / c_max * (x1 - x0)

    def Y(pi_atm):
        return y0 - pi_atm / pi_max * (y0 - y1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="34" fill="#e6edf3" font-size="18">Osmotic pressure vs concentration</text>',
        f'<text x="20" y="52" fill="#8b949e" font-size="12">'
        f'Pi = i c R T at 25 C -- more dissociated particles push harder</text>',
    ]

    # axes
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#8b949e" stroke-width="1.5"/>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#8b949e" stroke-width="1.5"/>')

    # y gridlines (atm)
    for pa in range(0, int(pi_max) + 1, 10):
        gy = Y(pa)
        parts.append(f'<line x1="{x0}" y1="{gy:.1f}" x2="{x1}" y2="{gy:.1f}" '
                     f'stroke="#21262d" stroke-width="1"/>')
        parts.append(f'<text x="{x0-8:.1f}" y="{gy+4:.1f}" fill="#8b949e" font-size="10" '
                     f'text-anchor="end">{pa} atm</text>')
    # x gridlines (mol/L)
    for cm in range(0, 8):
        c = cm * 100.0
        gx = X(c)
        parts.append(f'<line x1="{gx:.1f}" y1="{y0:.1f}" x2="{gx:.1f}" y2="{y0+4:.1f}" stroke="#8b949e"/>')
        parts.append(f'<text x="{gx:.1f}" y="{y0+16:.1f}" fill="#8b949e" font-size="10" '
                     f'text-anchor="middle">{c/1000:.1f} M</text>')

    # curves
    for label, i, col in solutes:
        pts = []
        for k in range(n + 1):
            c = c_max * k / n
            pts.append(f"{X(c):.1f},{Y(osmotic_pressure(c, T, i)/ATM):.1f}")
        parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{col}" stroke-width="2.6"/>')
        # label at right end
        yend = Y(osmotic_pressure(c_max, T, i) / ATM)
        parts.append(f'<text x="{x1-4:.1f}" y="{yend-4:.1f}" fill="{col}" font-size="11" '
                     f'text-anchor="end">{label}</text>')

    # markers: seawater (~27 atm) and blood (~7.6 atm) on the NaCl curve context
    for c, i, pi_atm, lbl, col in ((600.0, 2.0, 28.5, "seawater ~27 atm", "#ff6b6b"),
                                   (300.0, 1.0, 7.6, "blood ~7.6 atm", "#ffd43b")):
        gx, gy = X(c), Y(pi_atm)
        parts.append(f'<circle cx="{gx:.1f}" cy="{gy:.1f}" r="5" fill="{col}"/>')
        parts.append(f'<text x="{gx+9:.1f}" y="{gy+4:.1f}" fill="{col}" font-size="11">{lbl}</text>')

    parts.append(f'<text x="{x0+6:.1f}" y="{y1-6:.1f}" fill="#8b949e" font-size="11">'
                 f'reverse osmosis must apply more than Pi to push solvent back out</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
