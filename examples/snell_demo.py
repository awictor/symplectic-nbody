"""Demo: Snell's law -- bending, trapping, and reflecting light.

Prints refraction angles and critical angles across media, then draws rays leaving water at
growing incidence: bending away from the normal until, past the critical angle, they flip to
total internal reflection -- the effect that guides light down an optical fibre.

    python examples/snell_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from snell import (refraction_angle, critical_angle, is_total_internal_reflection,  # noqa: E402
                   brewster_angle, numerical_aperture, acceptance_angle,
                   N_AIR, N_WATER, N_GLASS, N_DIAMOND)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Snell's law n1 sin(t1) = n2 sin(t2): refraction, TIR, and Brewster\n")
    print(f"  {'medium':<12}{'n':>7}{'critical angle':>18}{'Brewster (from air)':>22}")
    for name, n in (("water", N_WATER), ("glass", N_GLASS), ("diamond", N_DIAMOND)):
        tc = math.degrees(critical_angle(n, N_AIR))
        tb = math.degrees(brewster_angle(N_AIR, n))
        print(f"  {name:<12}{n:>7.3f}{tc:>15.1f} deg{tb:>18.1f} deg")

    print("\n  Diamond's tiny 24 deg critical angle traps light through many internal bounces")
    print("  before it escapes -- that repeated total internal reflection is the sparkle.\n")

    print("  Air -> water refraction (light bends toward the normal entering denser water):")
    for inc in (10, 30, 50, 70):
        t2 = math.degrees(refraction_angle(N_AIR, math.radians(inc), N_WATER))
        print(f"    incidence {inc:>2} deg  ->  refracts to {t2:.1f} deg")

    NA = numerical_aperture(1.48, 1.46)
    print("\n  Optical fibre (core 1.48, clad 1.46): NA = %.3f, acceptance cone half-angle %.1f deg."
          % (NA, math.degrees(acceptance_angle(1.48, 1.46))))
    print("  Light inside that cone is trapped by total internal reflection and guided for km.")

    _svg(os.path.join(outdir, "snell.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'snell.svg')}")


def _svg(path, size=720, pad=70):
    # Rays inside water hitting the water-air surface at growing incidence; show refraction
    # bending away, then TIR past the critical angle.
    n1, n2 = N_WATER, N_AIR
    tc = critical_angle(n1, n2)

    cx = size * 0.5
    surf_y = size * 0.46
    x0, x1 = pad, size - pad

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="34" fill="#e6edf3" font-size="18">Refraction and total internal reflection</text>',
        f'<text x="20" y="52" fill="#8b949e" font-size="12">'
        f'rays leaving water bend away from the normal, then flip to TIR past the critical angle</text>',
    ]

    # water (bottom) / air (top)
    parts.append(f'<rect x="0" y="{surf_y:.1f}" width="{size}" height="{size-surf_y:.1f}" '
                 f'fill="#4dabf7" opacity="0.10"/>')
    parts.append(f'<line x1="0" y1="{surf_y:.1f}" x2="{size}" y2="{surf_y:.1f}" '
                 f'stroke="#8b949e" stroke-width="2"/>')
    parts.append(f'<text x="{x1-4:.1f}" y="{surf_y-8:.1f}" fill="#8b949e" font-size="11" text-anchor="end">air (n=1.00)</text>')
    parts.append(f'<text x="{x1-4:.1f}" y="{surf_y+18:.1f}" fill="#4dabf7" font-size="11" text-anchor="end">water (n=1.333)</text>')
    # normal
    parts.append(f'<line x1="{cx:.1f}" y1="{surf_y-140:.1f}" x2="{cx:.1f}" y2="{surf_y+140:.1f}" '
                 f'stroke="#21262d" stroke-width="1" stroke-dasharray="4 4"/>')

    incidences = [20, 40, 48.6, 60]     # deg (48.6 ~ critical)
    Lin = 150
    for inc in incidences:
        thi = math.radians(inc)
        # incoming ray from lower-left up to the origin
        ix = cx - Lin * math.sin(thi)
        iy = surf_y + Lin * math.cos(thi)
        col = "#06d6a0" if inc < math.degrees(tc) - 0.5 else "#ff6b6b"
        parts.append(f'<line x1="{ix:.1f}" y1="{iy:.1f}" x2="{cx:.1f}" y2="{surf_y:.1f}" '
                     f'stroke="{col}" stroke-width="2"/>')
        if is_total_internal_reflection(n1, thi, n2):
            # reflected ray back into water (mirror about normal)
            rx = cx + Lin * math.sin(thi)
            ry = surf_y + Lin * math.cos(thi)
            parts.append(f'<line x1="{cx:.1f}" y1="{surf_y:.1f}" x2="{rx:.1f}" y2="{ry:.1f}" '
                         f'stroke="#ff6b6b" stroke-width="2" stroke-dasharray="5 3"/>')
        else:
            t2 = refraction_angle(n1, thi, n2)
            rx = cx + Lin * math.sin(t2)
            ry = surf_y - Lin * math.cos(t2)
            parts.append(f'<line x1="{cx:.1f}" y1="{surf_y:.1f}" x2="{rx:.1f}" y2="{ry:.1f}" '
                         f'stroke="#06d6a0" stroke-width="2"/>')

    # critical-angle marker
    parts.append(f'<text x="{cx-Lin*math.sin(tc)-6:.1f}" y="{surf_y+Lin*math.cos(tc)+16:.1f}" '
                 f'fill="#ffd43b" font-size="11" text-anchor="end">critical {math.degrees(tc):.1f} deg</text>')

    parts.append(f'<text x="{x0:.1f}" y="{size-26:.1f}" fill="#06d6a0" font-size="11">'
                 f'green = refracts out</text>')
    parts.append(f'<text x="{x1:.1f}" y="{size-26:.1f}" fill="#ff6b6b" font-size="11" text-anchor="end">'
                 f'red = totally reflected (trapped)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
