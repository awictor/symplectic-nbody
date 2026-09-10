"""Demo: the Strouhal number -- the beat of a von Karman vortex street.

Prints vortex-shedding frequencies for wires, antennas and chimneys, the Roshko rise of St
with Reynolds number, and the lock-in wind speed, then draws a von Karman street: alternating
vortices peeling off a cylinder into the staggered double row that hums in the wind.

    python examples/strouhal_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from strouhal import (shedding_frequency, roshko_strouhal, lock_in_velocity,  # noqa: E402
                      vortex_spacing, ST_CYLINDER)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Strouhal number St = f d / U ~ 0.2: the rhythm of vortex shedding\n")
    print(f"  {'body':<28}{'d':>9}{'wind':>9}{'shed freq':>12}")
    cases = [
        ("telephone wire", 0.005, 10.0),
        ("car antenna", 0.008, 25.0),
        ("ship rigging", 0.02, 15.0),
        ("factory chimney", 3.0, 12.0),
    ]
    for label, d, U in cases:
        f = shedding_frequency(U, d)
        fs = f"{f:.1f} Hz" if f >= 1 else f"{f*1000:.1f} mHz"
        print(f"  {label:<28}{d*1000:>7.0f}mm{U:>7.0f} m/s{fs:>12}")

    print("\n  Roshko St = 0.212(1 - 21.2/Re) rises toward ~0.21 as Re grows:")
    for Re in (100, 300, 1000, 1e4, 1e6):
        print(f"    Re = {Re:>8.0f}  ->  St = {roshko_strouhal(Re):.3f}")

    d_ch, fn = 3.0, 0.25
    U_lock = lock_in_velocity(fn, d_ch)
    print("\n  A 3 m chimney with a 0.25 Hz sway mode locks in at wind ~%.1f m/s -- when the" % U_lock)
    print("  shedding beat hits the structure's resonance, the alternating side-force can")
    print("  build destructive vortex-induced vibration (why chimneys wear helical strakes).")
    print("  Wake vortices trail ~%.0f diameters apart." % (vortex_spacing(10.0, d_ch) / d_ch))

    _svg(os.path.join(outdir, "strouhal.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'strouhal.svg')}")


def _svg(path, size=720, pad=60):
    x0, x1 = pad, size - pad
    cyl_x = x0 + 60
    axis = size * 0.42
    d_px = 34                                  # cylinder diameter in px
    r = d_px / 2

    # vortex street: two staggered rows drifting downstream
    lam = d_px / ST_CYLINDER                   # wake wavelength ~ d/St ~ 5 d
    n_pairs = int((x1 - cyl_x - 40) / lam)
    off = axis - d_px * 0.9                     # top row offset
    off_b = axis + d_px * 0.9

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="34" fill="#e6edf3" font-size="18">The von Karman vortex street</text>',
        f'<text x="20" y="52" fill="#8b949e" font-size="12">'
        f'a cylinder sheds vortices alternately at f = St U / d, St ~ 0.2 -- the staggered wake that hums</text>',
    ]

    # incoming flow arrows
    for yy in (axis - 60, axis, axis + 60):
        parts.append(f'<line x1="{x0-20:.1f}" y1="{yy:.1f}" x2="{cyl_x-r-6:.1f}" y2="{yy:.1f}" '
                     f'stroke="#8b949e" stroke-width="1.4"/>')
        parts.append(f'<polygon points="{cyl_x-r-6:.1f},{yy:.1f} {cyl_x-r-16:.1f},{yy-4:.1f} '
                     f'{cyl_x-r-16:.1f},{yy+4:.1f}" fill="#8b949e"/>')
    parts.append(f'<text x="{x0-16:.1f}" y="{axis-70:.1f}" fill="#8b949e" font-size="11">flow U</text>')

    # the cylinder
    parts.append(f'<circle cx="{cyl_x:.1f}" cy="{axis:.1f}" r="{r:.1f}" fill="#30363d" '
                 f'stroke="#e6edf3" stroke-width="2"/>')

    # top row (counter-clockwise, blue) and bottom row (clockwise, red), staggered by lam/2
    for k in range(n_pairs):
        xt = cyl_x + r + 30 + k * lam
        xb = xt + lam / 2.0
        rv = r * (0.75 + 0.15 * math.cos(k))    # slight size variation, deterministic
        if xt < x1:
            parts.append(f'<circle cx="{xt:.1f}" cy="{off:.1f}" r="{rv:.1f}" fill="none" '
                         f'stroke="#4dabf7" stroke-width="2" opacity="0.85"/>')
            parts.append(f'<path d="M {xt-rv:.1f} {off:.1f} A {rv:.1f} {rv:.1f} 0 0 1 '
                         f'{xt:.1f} {off-rv:.1f}" fill="none" stroke="#4dabf7" stroke-width="2.6"/>')
        if xb < x1:
            parts.append(f'<circle cx="{xb:.1f}" cy="{off_b:.1f}" r="{rv:.1f}" fill="none" '
                         f'stroke="#ff6b6b" stroke-width="2" opacity="0.85"/>')
            parts.append(f'<path d="M {xb+rv:.1f} {off_b:.1f} A {rv:.1f} {rv:.1f} 0 0 1 '
                         f'{xb:.1f} {off_b+rv:.1f}" fill="none" stroke="#ff6b6b" stroke-width="2.6"/>')

    # wavelength marker
    xa = cyl_x + r + 30
    parts.append(f'<line x1="{xa:.1f}" y1="{axis+d_px*2.1:.1f}" x2="{xa+lam:.1f}" y2="{axis+d_px*2.1:.1f}" '
                 f'stroke="#ffd43b" stroke-width="1.4"/>')
    parts.append(f'<text x="{xa+lam/2:.1f}" y="{axis+d_px*2.1+16:.1f}" fill="#ffd43b" '
                 f'font-size="11" text-anchor="middle">wake wavelength lambda = d/St ~ 5d</text>')

    # frequency-vs-windspeed inset (bottom)
    py0, py1 = size * 0.74, size - pad
    px0, px1 = x0, x1
    d_ref = 0.02
    Umax = 40.0
    fmax = shedding_frequency(Umax, d_ref)
    def FX(U):
        return px0 + U / Umax * (px1 - px0)
    def FY(f):
        return py1 - f / fmax * (py1 - py0)
    parts.append(f'<line x1="{px0}" y1="{py1}" x2="{px1}" y2="{py1}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{px0}" y1="{py1}" x2="{px0}" y2="{py0}" stroke="#8b949e" stroke-width="1.2"/>')
    fpts = " ".join(f"{FX(U):.1f},{FY(shedding_frequency(U, d_ref)):.1f}" for U in range(0, 41))
    parts.append(f'<polyline points="{fpts}" fill="none" stroke="#06d6a0" stroke-width="2.4"/>')
    parts.append(f'<text x="{px0+6:.1f}" y="{py0+12:.1f}" fill="#06d6a0" font-size="11">'
                 f'shed frequency vs wind speed (20 mm rod): linear, f = St U/d</text>')
    parts.append(f'<text x="{px1-4:.1f}" y="{py1-6:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="end">40 m/s</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
