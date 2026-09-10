"""Demo: Milankovitch cycles -- orbital pacing of the ice ages.

Prints how 65N midsummer insolation (the ice-sheet control knob) shifts as eccentricity,
obliquity and precession change, then draws the seasonal insolation map: daily top-of-
atmosphere sunlight over latitude (y) and time of year (x), with the polar day/night regions
and the June/December solstice peaks.

    python examples/milankovitch_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from milankovitch import (daily_insolation, summer_solstice_insolation,  # noqa: E402
                          climatic_precession, OBLIQUITY_NOW)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    phi65 = math.radians(65.0)
    print("Milankovitch: 65N midsummer insolation drives ice sheets (grow when it's weak)\n")
    print("  Present orbit: e=0.0167, obliquity=23.44 deg, perihelion in NH winter\n")
    print(f"  {'orbital state':<40}{'65N June (W/m^2)':>18}")
    cases = [
        ("present day", 0.0167, 23.44, 283.0),
        ("low obliquity 22.1 deg (cool summers)", 0.0167, 22.1, 283.0),
        ("high obliquity 24.5 deg (warm summers)", 0.0167, 24.5, 283.0),
        ("high ecc 0.05, summer at aphelion", 0.05, 23.44, 270.0),
        ("high ecc 0.05, summer at perihelion", 0.05, 23.44, 90.0),
    ]
    base = None
    for label, e, eps_deg, omega_deg in cases:
        q = summer_solstice_insolation(phi65, e, math.radians(eps_deg),
                                       math.radians(omega_deg))
        if base is None:
            base = q
        cp = climatic_precession(e, math.radians(omega_deg))
        print(f"  {label:<40}{q:>13.1f}   ({q-base:+.1f})")

    print("\n  Cool NH summers (low obliquity + summer at aphelion) let winter snow survive")
    print("  and ice sheets grow -- the ~41 kyr tilt and ~23 kyr precession beats, modulated")
    print("  by the ~100 kyr eccentricity envelope, that pace the Pleistocene glacial cycles.")

    _svg(os.path.join(outdir, "milankovitch.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'milankovitch.svg')}")


def _color(q, qmax):
    # dark-blue (low) -> yellow (high) ramp on a 0..qmax scale
    t = max(0.0, min(1.0, q / qmax))
    if t == 0.0:
        return "#0d1117"                       # polar night: background
    # blend #1b3a6b (cold) -> #4dabf7 (blue) -> #ffd43b (yellow)
    if t < 0.5:
        u = t / 0.5
        r = int(0x1b + u * (0x4d - 0x1b))
        g = int(0x3a + u * (0xab - 0x3a))
        b = int(0x6b + u * (0xf7 - 0x6b))
    else:
        u = (t - 0.5) / 0.5
        r = int(0x4d + u * (0xff - 0x4d))
        g = int(0xab + u * (0xd4 - 0xab))
        b = int(0xf7 + u * (0x3b - 0xf7))
    return f"#{r:02x}{g:02x}{b:02x}"


def _svg(path, size=720, pad=72):
    # insolation map: x = solar longitude (season) 0..360, y = latitude -90..90
    nx, ny = 96, 72
    e, eps, omega = 0.0167, OBLIQUITY_NOW, math.radians(283.0)
    grid = []
    qmax = 0.0
    for j in range(ny):
        phi = math.radians(-90.0 + 180.0 * (j + 0.5) / ny)
        row = []
        for i in range(nx):
            lam = 2 * math.pi * (i + 0.5) / nx
            q = daily_insolation(phi, lam, e, eps, omega)
            row.append(q)
            qmax = max(qmax, q)
        grid.append(row)

    x0, y0 = pad, pad + 30
    w, h = size - 2 * pad, size - 2 * pad - 30
    cw, ch = w / nx, h / ny

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="34" fill="#e6edf3" font-size="18">Seasonal insolation map</text>',
        f'<text x="20" y="52" fill="#8b949e" font-size="12">'
        f'daily top-of-atmosphere sunlight (W/m^2) by latitude and time of year</text>',
    ]

    for j in range(ny):
        # y grows downward = latitude decreasing from +90 at top
        yy = y0 + j * ch
        for i in range(nx):
            q = grid[j][i]
            col = _color(q, qmax)
            if col == "#0d1117":
                continue      # skip polar-night cells (leave background)
            xx = x0 + i * cw
            parts.append(f'<rect x="{xx:.1f}" y="{yy:.1f}" width="{cw+0.6:.1f}" '
                         f'height="{ch+0.6:.1f}" fill="{col}"/>')

    # frame
    parts.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{w:.1f}" height="{h:.1f}" '
                 f'fill="none" stroke="#8b949e" stroke-width="1.5"/>')

    # latitude ticks
    for lat in (90, 45, 0, -45, -90):
        yy = y0 + (90 - lat) / 180.0 * h
        parts.append(f'<text x="{x0-8:.1f}" y="{yy+4:.1f}" fill="#8b949e" font-size="10" '
                     f'text-anchor="end">{lat:>3} deg</text>')
        parts.append(f'<line x1="{x0-4:.1f}" y1="{yy:.1f}" x2="{x0:.1f}" y2="{yy:.1f}" stroke="#8b949e"/>')
    # equator line
    yeq = y0 + 0.5 * h
    parts.append(f'<line x1="{x0:.1f}" y1="{yeq:.1f}" x2="{x0+w:.1f}" y2="{yeq:.1f}" '
                 f'stroke="#8b949e" stroke-width="0.8" stroke-dasharray="4 4" opacity="0.5"/>')

    # season ticks
    for lam_deg, lbl in ((0, "Mar eq"), (90, "Jun sol"), (180, "Sep eq"), (270, "Dec sol")):
        xx = x0 + lam_deg / 360.0 * w
        parts.append(f'<text x="{xx:.1f}" y="{y0+h+16:.1f}" fill="#8b949e" font-size="10" '
                     f'text-anchor="middle">{lbl}</text>')
        parts.append(f'<line x1="{xx:.1f}" y1="{y0+h:.1f}" x2="{xx:.1f}" y2="{y0+h+4:.1f}" stroke="#8b949e"/>')

    # mark the 65N June target
    xj = x0 + 90.0 / 360.0 * w
    yj = y0 + (90 - 65) / 180.0 * h
    parts.append(f'<circle cx="{xj:.1f}" cy="{yj:.1f}" r="5" fill="none" stroke="#ff6b6b" stroke-width="2"/>')
    parts.append(f'<text x="{xj+9:.1f}" y="{yj-6:.1f}" fill="#ff6b6b" font-size="11">65N June (ice-sheet knob)</text>')

    parts.append(f'<text x="{x0:.1f}" y="{y0-8:.1f}" fill="#8b949e" font-size="11">'
                 f'dark wedges top-left/bottom-right = polar night; bright bands = midnight sun</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
