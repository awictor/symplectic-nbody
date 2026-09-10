"""Demo: sound speed and the Rankine-Hugoniot shock jumps.

Prints the jump ratios across shocks of increasing Mach number, then draws the
density, pressure and temperature jumps vs Mach on a log axis -- density flattening
at 4 while pressure and temperature climb without bound.

    python examples/shock_jump_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from shock_jump import (sound_speed, density_ratio, pressure_ratio,  # noqa: E402
                        temperature_ratio, downstream_mach,
                        strong_shock_compression)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Rankine-Hugoniot shock jumps (gamma = 5/3)\n")
    print(f"  sound speed, ionized gas at 1e4 K: {sound_speed(1e4)/1e3:.1f} km/s")
    print(f"  sound speed, air at 288 K:         {sound_speed(288, mu=28.97, gamma=1.4):.0f} m/s\n")
    print(f"  {'Mach':>7}{'rho2/rho1':>12}{'P2/P1':>12}{'T2/T1':>12}{'M2 (down)':>12}")
    print("  " + "-" * 55)
    for M in (1.0, 1.5, 2, 3, 5, 10, 30, 100):
        print(f"  {M:>7g}{density_ratio(M):>12.3f}{pressure_ratio(M):>12.1f}"
              f"{temperature_ratio(M):>12.1f}{downstream_mach(M):>12.3f}")

    print(f"\n  Density compression saturates at (gamma+1)/(gamma-1) = "
          f"{strong_shock_compression():.0f} -- a strong shock can pack gas only")
    print("  fourfold. But pressure and temperature jumps grow as M^2 without limit,")
    print("  which is why strong shocks heat gas to millions of kelvin (supernova")
    print("  remnants, re-entry plasma) while barely compressing it. The downstream")
    print("  flow is always subsonic (M2 < 1) -- the shock is a one-way valve.")

    _svg(os.path.join(outdir, "shock_jump.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'shock_jump.svg')}")


def _svg(path, size=720, pad=68):
    machs = [1.0 + 0.1 * i for i in range(0, 140)]   # 1 .. ~14.9
    series = [
        ("rho2/rho1", [density_ratio(M) for M in machs], "#4dabf7"),
        ("P2/P1", [pressure_ratio(M) for M in machs], "#ff922b"),
        ("T2/T1", [temperature_ratio(M) for M in machs], "#ff6b6b"),
    ]
    lx = [math.log10(M) for M in machs]
    all_y = [y for _, ys, _ in series for y in ys]
    ly_all = [math.log10(y) for y in all_y]
    xmin, xmax = lx[0], lx[-1]
    ymin, ymax = min(ly_all), max(ly_all)

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
    # r=4 saturation line
    y4 = sy(math.log10(4.0))
    parts.append(f'<line x1="{pad}" y1="{y4:.1f}" x2="{size-pad}" y2="{y4:.1f}" '
                 f'stroke="#4dabf7" stroke-width="1" stroke-dasharray="3 4" opacity="0.6"/>')
    parts.append(f'<text x="{pad+8}" y="{y4-5:.1f}" fill="#4dabf7" '
                 f'font-size="10">density ceiling = 4</text>')

    ytop = pad + 22
    for i, (label, ys, col) in enumerate(series):
        ly = [math.log10(y) for y in ys]
        poly = " ".join(f"{sx(lx[j]):.1f},{sy(ly[j]):.1f}" for j in range(len(machs)))
        parts.append(f'<polyline points="{poly}" fill="none" stroke="{col}" stroke-width="2.4"/>')
        parts.append(f'<text x="{pad+10}" y="{ytop + i*16}" fill="{col}" '
                     f'font-size="12">{label}</text>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Rankine-Hugoniot jumps vs Mach number</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'density saturates at 4; pressure and temperature diverge as M^2</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+24}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">log10 Mach number -&gt;</text>')
    parts.append(f'<text x="{pad-14}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'log10 jump ratio</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
