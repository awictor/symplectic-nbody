"""Demo: atmospheric Jeans escape -- the cosmic shoreline of who keeps an air.

Prints escape parameters for gases on several worlds, then draws the classic
retention diagram: escape speed vs molecular thermal speed, with the v_esc = 6 v_th
retention line that separates worlds/gases that hold on from those that leak away.

    python examples/jeans_escape_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from jeans_escape import (escape_speed, thermal_speed, escape_parameter,  # noqa: E402
                          is_retained, RETENTION_RATIO, M_EARTH, R_EARTH,
                          M_MOON, R_MOON, M_H2, M_HE, M_N2, M_O2, M_CO2, M_H2O)

# (name, mass kg, radius m, exobase T K)
BODIES = [
    ("Earth", M_EARTH, R_EARTH, 1000.0),
    ("Moon", M_MOON, R_MOON, 400.0),
    ("Mars", 6.417e23, 3.390e6, 350.0),
    ("Titan", 1.345e23, 2.575e6, 180.0),
    ("Jupiter", 1.898e27, 6.9911e7, 1000.0),
]
GASES = [("H2", M_H2), ("He", M_HE), ("H2O", M_H2O),
         ("N2", M_N2), ("O2", M_O2), ("CO2", M_CO2)]


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Jeans escape: lambda = v_esc^2 / v_th^2 = G M m / (R k_B T)\n")
    print(f"  retention rule of thumb: keep a gas when v_esc >= {RETENTION_RATIO:.0f} v_th"
          f" (lambda >= {RETENTION_RATIO**2:.0f})\n")
    header = f"  {'body':>8}{'v_esc':>9}" + "".join(f"{g[0]:>7}" for g in GASES)
    print(header + "   (Y=kept, .=lost)")
    print("  " + "-" * (len(header) - 2 + 8))
    for bname, M, R, T in BODIES:
        ve = escape_speed(M, R) / 1e3
        row = f"  {bname:>8}{ve:>8.1f}k"
        for _, m in GASES:
            row += f"{('Y' if is_retained(M, R, T, m) else '.'):>7}"
        print(row)

    print("\n  Earth keeps N2/O2/CO2/water but loses H2 and He (why our air is heavy);")
    print("  the hot, low-gravity Moon loses all but the heaviest (and even CO2 goes")
    print("  to non-thermal escape); cold Titan clings even to N2;")
    print("  giant Jupiter keeps everything, including hydrogen. This 'cosmic")
    print("  shoreline' -- escape speed vs temperature -- sorts which worlds have air.")

    _svg(os.path.join(outdir, "jeans_escape.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'jeans_escape.svg')}")


def _svg(path, size=720, pad=70):
    # x = thermal speed (km/s, log), y = escape speed (km/s, log)
    pts = []
    for bname, M, R, T in BODIES:
        ve = escape_speed(M, R) / 1e3
        for gname, m in GASES:
            vth = thermal_speed(T, m) / 1e3
            pts.append((bname, gname, vth, ve, is_retained(M, R, T, m)))
    xs = [p[2] for p in pts]
    ys = [p[3] for p in pts]
    lx = [math.log10(x) for x in xs]
    ly = [math.log10(y) for y in ys]
    xmin, xmax = min(lx) - 0.1, max(lx) + 0.1
    ymin, ymax = min(ly) - 0.1, max(ly) + 0.1

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
    # retention line v_esc = ratio * v_th  ->  log ve = log vth + log ratio
    lr = math.log10(RETENTION_RATIO)
    x1, x2 = xmin, xmax
    parts.append(f'<line x1="{sx(x1):.1f}" y1="{sy(x1+lr):.1f}" '
                 f'x2="{sx(x2):.1f}" y2="{sy(x2+lr):.1f}" '
                 f'stroke="#ff6b6b" stroke-width="1.4" stroke-dasharray="5 4"/>')
    parts.append(f'<text x="{sx(x2)-6:.1f}" y="{sy(x2+lr)-6:.1f}" fill="#ff6b6b" '
                 f'font-size="11" text-anchor="end">v_esc = 6 v_th (retention line)</text>')

    for bname, gname, vth, ve, kept in pts:
        px = sx(math.log10(vth))
        py = sy(math.log10(ve))
        col = "#06d6a0" if kept else "#ff6b6b"
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.5" fill="{col}"/>')
        parts.append(f'<text x="{px+5:.1f}" y="{py+3:.1f}" fill="#8b949e" '
                     f'font-size="8">{bname[:2]}:{gname}</text>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'The cosmic shoreline: escape vs thermal speed</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'green = retained over geologic time, red = escapes</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+24}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">log10 thermal speed (km/s) -&gt;</text>')
    parts.append(f'<text x="{pad-12}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'log10 escape speed (km/s)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
