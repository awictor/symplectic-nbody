"""Demo: Hohmann vs bi-elliptic transfers.

Prints the total delta-v of the Hohmann and bi-elliptic transfers across radius ratios,
then draws both (normalized to circular speed) vs the ratio R = r2/r1, with the classic
R = 11.94 crossover where the three-burn detour starts to win.

    python examples/bi_elliptic_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bi_elliptic import (hohmann_delta_v, bi_elliptic_delta_v,  # noqa: E402
                         bi_elliptic_is_cheaper, crossover_ratio, MU_EARTH)

R1 = 7000e3


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    v1 = math.sqrt(MU_EARTH / R1)
    print("Hohmann vs bi-elliptic transfer (r_b = 100 r1 detour)\n")
    print(f"  crossover ratio R = r2/r1 = {crossover_ratio():.2f}\n")
    print(f"  {'R = r2/r1':>12}{'Hohmann dv':>13}{'bi-elliptic':>13}{'winner':>13}")
    print("  " + "-" * 51)
    for R in (5, 10, 11.94, 13, 16, 30, 60):
        r2 = R * R1
        rb = 100 * R1 if R * R1 < 100 * R1 else 3 * R * R1
        rb = max(rb, 1.5 * r2)
        dvh = hohmann_delta_v(R1, r2)
        dvb = bi_elliptic_delta_v(R1, r2, rb)
        winner = "bi-elliptic" if dvb < dvh else "Hohmann"
        print(f"  {R:>12.2f}{dvh:>12.0f}m{dvb:>12.0f}m{winner:>13}")

    print("\n  For modest orbit changes the Hohmann two-burn is unbeatable. Past R ~ 12,")
    print("  flinging the craft far beyond the target and dropping back costs less")
    print("  total delta-v, because the mid-course burn happens where orbital speeds")
    print("  are tiny. The catch: the detour can take years, so bi-elliptic is used")
    print("  only for the most extreme orbit raises. The 11.94 crossover is exact.")

    _svg(os.path.join(outdir, "bi_elliptic.svg"), v1)
    print(f"\n  wrote {os.path.join(outdir, 'bi_elliptic.svg')}")


def _svg(path, v1, size=720, pad=72):
    Rs = [2.0 + 0.6 * i for i in range(0, 100)]   # 2 .. ~62
    dvh = [hohmann_delta_v(R1, R * R1) / v1 for R in Rs]
    # bi-elliptic with a generous detour r_b = 50 r2 (near the limiting case)
    dvb = [bi_elliptic_delta_v(R1, R * R1, 50 * R * R1) / v1 for R in Rs]
    xmin, xmax = Rs[0], Rs[-1]
    allv = dvh + dvb
    ymin, ymax = min(allv) * 0.98, max(allv) * 1.02

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
    # crossover vertical line
    xc = sx(crossover_ratio())
    parts.append(f'<line x1="{xc:.1f}" y1="{pad}" x2="{xc:.1f}" y2="{size-pad}" '
                 f'stroke="#8b949e" stroke-width="1.2" stroke-dasharray="4 4"/>')
    parts.append(f'<text x="{xc+5:.1f}" y="{pad+40:.1f}" fill="#8b949e" '
                 f'font-size="11">R = 11.94 crossover</text>')

    ph = " ".join(f"{sx(Rs[i]):.1f},{sy(dvh[i]):.1f}" for i in range(len(Rs)))
    pb = " ".join(f"{sx(Rs[i]):.1f},{sy(dvb[i]):.1f}" for i in range(len(Rs)))
    parts.append(f'<polyline points="{ph}" fill="none" stroke="#4dabf7" stroke-width="2.4"/>')
    parts.append(f'<polyline points="{pb}" fill="none" stroke="#ff922b" stroke-width="2.4"/>')

    parts.append(f'<text x="{pad+10}" y="{pad+22}" fill="#4dabf7" font-size="12">'
                 f'Hohmann (2 burns)</text>')
    parts.append(f'<text x="{pad+10}" y="{pad+38}" fill="#ff922b" font-size="12">'
                 f'bi-elliptic (3 burns, r_b = 50 r2)</text>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Transfer delta-v vs orbit radius ratio</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'bi-elliptic dips below Hohmann past R ~ 12</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+24}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">radius ratio R = r2 / r1 -&gt;</text>')
    parts.append(f'<text x="{pad-18}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'total delta-v / circular speed</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
