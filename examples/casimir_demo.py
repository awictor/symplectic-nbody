"""Demo: the Casimir effect -- two plates pushed together by empty space.

Prints the Casimir pressure and force across plate gaps and the gap at which it matches
atmospheric pressure, then draws pressure versus separation on log-log axes -- the steep
d^-4 law, negligible at macroscopic gaps but crushing below 100 nm where it causes MEMS
stiction.

    python examples/casimir_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from casimir import (casimir_pressure, casimir_force, separation_for_pressure)  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Casimir effect: vacuum pushes two plates together, P = pi^2 hbar c / (240 d^4)\n")
    print(f"  {'gap d':>10}{'pressure':>16}{'force on 1 cm^2':>18}")
    for d in (10e-9, 100e-9, 1e-6, 10e-6, 1e-3):
        P = casimir_pressure(d)
        F = casimir_force(1e-4, d)
        ps = f"{P:.2e} Pa"
        fs = f"{F*1e6:.3f} uN" if F > 1e-9 else f"{F:.2e} N"
        print(f"  {_len(d):>10}{ps:>16}{fs:>18}")

    d_atm = separation_for_pressure(101325.0)
    print("\n  At d = %.1f nm the Casimir pressure equals one atmosphere. The force is pure" % (d_atm * 1e9))
    print("  quantum vacuum -- only the modes that fit between the plates survive inside, so")
    print("  the fuller outside vacuum presses them together. The d^-4 law makes it invisible")
    print("  at human scales but dominant in MEMS, where it sticks micro-parts together (stiction).")

    _svg(os.path.join(outdir, "casimir.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'casimir.svg')}")


def _len(d):
    if d < 1e-6:
        return f"{d*1e9:.0f} nm"
    if d < 1e-3:
        return f"{d*1e6:.0f} um"
    return f"{d*1e3:.0f} mm"


def _svg(path, size=720, pad=82):
    # log-log Casimir pressure vs gap, 1 nm .. 10 um
    ds = [10 ** (-9 + (math.log10(1e-5) - (-9)) * i / 199) for i in range(200)]
    ps = [casimir_pressure(d) for d in ds]

    lx0, lx1 = math.log10(ds[0]), math.log10(ds[-1])
    ly0, ly1 = math.log10(min(ps)), math.log10(max(ps))

    x0, x1 = pad, size - pad
    y0, y1 = size - pad, pad + 44

    def X(d):
        return x0 + (math.log10(d) - lx0) / (lx1 - lx0) * (x1 - x0)

    def Y(p):
        return y0 - (math.log10(p) - ly0) / (ly1 - ly0) * (y0 - y1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="34" fill="#e6edf3" font-size="18">The Casimir pressure of the vacuum</text>',
        f'<text x="20" y="52" fill="#8b949e" font-size="12">'
        f'P ~ d^-4: nothing at human gaps, an atmosphere by ~10 nm</text>',
    ]

    # axes
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#8b949e" stroke-width="1.5"/>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#8b949e" stroke-width="1.5"/>')

    # pressure decade gridlines/labels
    for e in range(int(math.floor(ly0)), int(math.ceil(ly1)) + 1, 2):
        p = 10.0 ** e
        gy = Y(p)
        if not (y1 <= gy <= y0):
            continue
        parts.append(f'<line x1="{x0}" y1="{gy:.1f}" x2="{x1}" y2="{gy:.1f}" '
                     f'stroke="#21262d" stroke-width="1"/>')
        parts.append(f'<text x="{x0-8:.1f}" y="{gy+4:.1f}" fill="#8b949e" font-size="10" '
                     f'text-anchor="end">10^{e} Pa</text>')
    # gap decade ticks
    for e in range(-9, -4):
        gx = X(10.0 ** e)
        parts.append(f'<line x1="{gx:.1f}" y1="{y0:.1f}" x2="{gx:.1f}" y2="{y0+4:.1f}" stroke="#8b949e"/>')
        lbl = f"{10**(e+9):.0f} nm" if e < -6 else f"{10**(e+6):.0f} um"
        parts.append(f'<text x="{gx:.1f}" y="{y0+16:.1f}" fill="#8b949e" font-size="10" '
                     f'text-anchor="middle">{lbl}</text>')
    parts.append(f'<text x="{(x0+x1)/2:.1f}" y="{y0+30:.1f}" fill="#8b949e" font-size="11" '
                 f'text-anchor="middle">plate separation d</text>')

    # atmospheric-pressure reference line
    gy_atm = Y(101325.0)
    if y1 <= gy_atm <= y0:
        parts.append(f'<line x1="{x0}" y1="{gy_atm:.1f}" x2="{x1}" y2="{gy_atm:.1f}" '
                     f'stroke="#ff6b6b" stroke-width="1" stroke-dasharray="5 4" opacity="0.7"/>')
        parts.append(f'<text x="{x1-4:.1f}" y="{gy_atm-5:.1f}" fill="#ff6b6b" font-size="10" '
                     f'text-anchor="end">1 atmosphere</text>')

    # the curve
    poly = " ".join(f"{X(ds[i]):.1f},{Y(ps[i]):.1f}" for i in range(len(ds)))
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#4dabf7" stroke-width="2.8"/>')

    # markers
    for d, col, lbl in ((100e-9, "#ffd43b", "100 nm: ~13 Pa"),
                        (separation_for_pressure(101325.0), "#ff6b6b", "= 1 atm"),
                        (1e-6, "#06d6a0", "1 um (MEMS)")):
        parts.append(f'<circle cx="{X(d):.1f}" cy="{Y(casimir_pressure(d)):.1f}" r="4.5" fill="{col}"/>')
        parts.append(f'<text x="{X(d)+7:.1f}" y="{Y(casimir_pressure(d))-6:.1f}" fill="{col}" '
                     f'font-size="10">{lbl}</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
