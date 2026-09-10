"""Demo: Poynting-Robertson drag and the fate of interplanetary dust.

Prints beta and inspiral time for a range of grain sizes, then draws inspiral time
vs grain size at several orbital radii, with the blow-out size and the age of the
solar system marked -- the window of sizes that spiral in rather than being blown
out or lasting forever.

    python examples/poynting_robertson_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from poynting_robertson import (beta, blowout_size, is_blown_out,  # noqa: E402
                                inspiral_time_years, AU, RHO_DUST)

AGE_SS_YEARS = 4.567e9


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Poynting-Robertson drag: dust's own re-radiated light is a headwind\n")
    sb = blowout_size() * 1e6
    print(f"  blow-out size (beta = 1/2): {sb:.2f} micron "
          f"(smaller grains are blown straight out)\n")
    print(f"  {'grain size':>12}{'beta':>9}{'fate / t_PR from 1 AU':>26}")
    print("  " + "-" * 48)
    for s_um in (0.1, 0.3, 0.5, 1.0, 10.0, 100.0, 1000.0):
        s = s_um * 1e-6
        b = beta(s)
        if is_blown_out(s):
            fate = "blown out (unbound)"
        else:
            fate = f"{inspiral_time_years(AU, s):,.0f} yr"
        label = f"{s_um:g} um" if s_um < 1000 else "1 mm"
        print(f"  {label:>12}{b:>9.3f}{fate:>26}")

    print("\n  Micron grains at 1 AU spiral into the Sun in a few thousand years --")
    print("  thousands of times shorter than the solar system's age. So the")
    print("  zodiacal dust cloud cannot be primordial; it is continuously")
    print("  resupplied by comet trails and asteroid collisions. Grains below the")
    print("  blow-out size never orbit at all -- they leave as beta meteoroids.")

    _svg(os.path.join(outdir, "poynting_robertson.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'poynting_robertson.svg')}")


def _svg(path, size=720, pad=68):
    sizes_um = [0.4 * (1.15 ** i) for i in range(0, 70)]
    sizes_um = [x for x in sizes_um if x <= 5000.0]
    radii = [(1.0, "#ff922b", "1 AU"), (3.0, "#4dabf7", "3 AU"),
             (30.0, "#b197fc", "30 AU")]

    curves = []
    for r_au, col, label in radii:
        ys = [inspiral_time_years(r_au * AU, x * 1e-6) for x in sizes_um]
        curves.append((ys, col, label))

    lx = [math.log10(x) for x in sizes_um]
    all_y = [y for ys, _, _ in curves for y in ys]
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
    # age of solar system horizontal line
    la = math.log10(AGE_SS_YEARS)
    if ymin <= la <= ymax:
        ya = sy(la)
        parts.append(f'<line x1="{pad}" y1="{ya:.1f}" x2="{size-pad}" y2="{ya:.1f}" '
                     f'stroke="#ff6b6b" stroke-width="1.2" stroke-dasharray="5 4"/>')
        parts.append(f'<text x="{pad+8}" y="{ya-6:.1f}" fill="#ff6b6b" '
                     f'font-size="11">age of the solar system</text>')
    # blow-out vertical line
    sb = blowout_size() * 1e6
    if xmin <= math.log10(sb) <= xmax:
        xb = sx(math.log10(sb))
        parts.append(f'<line x1="{xb:.1f}" y1="{pad}" x2="{xb:.1f}" y2="{size-pad}" '
                     f'stroke="#8b949e" stroke-width="1.2" stroke-dasharray="3 3"/>')
        parts.append(f'<text x="{xb+4:.1f}" y="{size-pad-8:.1f}" fill="#8b949e" '
                     f'font-size="10">blow-out {sb:.2f} um</text>')

    for ys, col, label in curves:
        ly = [math.log10(y) for y in ys]
        poly = " ".join(f"{sx(lx[i]):.1f},{sy(ly[i]):.1f}" for i in range(len(sizes_um)))
        parts.append(f'<polyline points="{poly}" fill="none" stroke="{col}" stroke-width="2.2"/>')

    ytop = pad + 20
    for i, (_, col, label) in enumerate(curves):
        parts.append(f'<text x="{size-pad-110}" y="{ytop + i*16}" fill="{col}" '
                     f'font-size="12">from {label}</text>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Poynting-Robertson inspiral time vs grain size</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f't_PR ~ r^2 s: small grains near the Sun vanish in millennia</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+24}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">log10 grain radius (micron) -&gt;</text>')
    parts.append(f'<text x="{pad-14}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'log10 inspiral time (yr)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
