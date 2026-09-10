"""Demo: the Sersic profile -- the shape of a galaxy's light.

Prints b_n, scale lengths and enclosed-light radii for the exponential disk (n=1), a bulge
(n=2), and the de Vaucouleurs law (n=4), then draws the surface-brightness profiles in
mag/arcsec^2 versus R/R_e -- the steep bright core and extended faint wings of a big n, all
crossing at the effective radius.

    python examples/sersic_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sersic import (b_n, surface_brightness, total_luminosity,  # noqa: E402
                    exponential_scale_length, enclosed_light_fraction, brightness_to_mag)


PROFILES = [
    ("n=1 exponential disk", 1.0, "#4dabf7"),
    ("n=2 bulge", 2.0, "#06d6a0"),
    ("n=4 de Vaucouleurs", 4.0, "#ff922b"),
]


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Sersic profile I(R) = I_e exp{-b_n[(R/R_e)^(1/n) - 1]}\n")
    print(f"  {'profile':<26}{'b_n':>8}{'I(0.1 R_e)/I_e':>16}{'R(90% light)/R_e':>18}")
    for label, n, _ in PROFILES:
        bn = b_n(n)
        core = surface_brightness(0.1, 1.0, 1.0, n)     # I_e=1, R_e=1
        r90 = _radius_for_fraction(0.9, n)
        print(f"  {label:<26}{bn:>8.3f}{core:>16.1f}{r90:>18.2f}")

    print("\n  n=1 exponential disk: scale length h = R_e/1.678 = %.3f R_e" % exponential_scale_length(1.0))
    print("  Half the light sits inside R_e for every n (that's the definition), but larger n")
    print("  packs a far brighter core AND flings more light into faint outer wings -- the")
    print("  cuspy, extended glow of a giant elliptical versus the gentle fade of a disk.")

    print("\n  Total luminosity L = I_e R_e^2 2 pi n e^b_n Gamma(2n)/b_n^2n (I_e=R_e=1):")
    for label, n, _ in PROFILES:
        print(f"    {label:<26} L = {total_luminosity(1.0, 1.0, n):.2f}")

    _svg(os.path.join(outdir, "sersic.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'sersic.svg')}")


def _radius_for_fraction(frac, n, r_e=1.0):
    lo, hi = 1e-3, 100.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if enclosed_light_fraction(mid, r_e, n) < frac:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _svg(path, size=720, pad=76):
    # Surface brightness in mag/arcsec^2 (mu_e = 21 at R_e) vs R/R_e on a log-radius axis.
    mu_e = 21.0
    rr = [10 ** (-1.3 + (0.9 - (-1.3)) * i / 199) for i in range(200)]   # 0.05 .. ~8 R_e
    lx0, lx1 = math.log10(rr[0]), math.log10(rr[-1])
    mu_lo, mu_hi = 16.0, 30.0        # brighter (small mu) at top

    x0, x1 = pad, size - pad
    y0, y1 = size - pad, pad + 40

    def X(r):
        return x0 + (math.log10(r) - lx0) / (lx1 - lx0) * (x1 - x0)

    def Y(mu):
        return y1 + (mu - mu_hi) / (mu_lo - mu_hi) * (y0 - y1)   # mu_lo(bright) at top=y1

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="34" fill="#e6edf3" font-size="18">Sersic surface-brightness profiles</text>',
        f'<text x="20" y="52" fill="#8b949e" font-size="12">'
        f'bigger n = brighter cusp and more extended wings; all cross at the effective radius R_e</text>',
    ]

    # axes
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#8b949e" stroke-width="1.5"/>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#8b949e" stroke-width="1.5"/>')
    # brightness gridlines (mag/arcsec^2)
    for mu in range(16, 31, 2):
        gy = Y(mu)
        parts.append(f'<line x1="{x0}" y1="{gy:.1f}" x2="{x1}" y2="{gy:.1f}" '
                     f'stroke="#21262d" stroke-width="1"/>')
        parts.append(f'<text x="{x0-8:.1f}" y="{gy+4:.1f}" fill="#8b949e" font-size="10" '
                     f'text-anchor="end">{mu}</text>')
    parts.append(f'<text x="22" y="{(y0+y1)/2:.1f}" fill="#8b949e" font-size="11" '
                 f'transform="rotate(-90 22 {(y0+y1)/2:.1f})" text-anchor="middle">mu (mag/arcsec^2, brighter up)</text>')
    # radius ticks (decades)
    for e in (-1, 0):
        gx = X(10.0 ** e)
        parts.append(f'<line x1="{gx:.1f}" y1="{y0:.1f}" x2="{gx:.1f}" y2="{y0+4:.1f}" stroke="#8b949e"/>')
        parts.append(f'<text x="{gx:.1f}" y="{y0+16:.1f}" fill="#8b949e" font-size="10" '
                     f'text-anchor="middle">{"0.1" if e==-1 else "1"} R_e</text>')

    # R_e marker
    parts.append(f'<line x1="{X(1.0):.1f}" y1="{y0:.1f}" x2="{X(1.0):.1f}" y2="{y1:.1f}" '
                 f'stroke="#ffd43b" stroke-width="1" stroke-dasharray="4 4" opacity="0.6"/>')
    parts.append(f'<circle cx="{X(1.0):.1f}" cy="{Y(mu_e):.1f}" r="4" fill="#ffd43b"/>')
    parts.append(f'<text x="{X(1.0)+6:.1f}" y="{Y(mu_e)-6:.1f}" fill="#ffd43b" font-size="11">'
                 f'R_e, mu_e = {mu_e:.0f}</text>')

    for label, n, col in PROFILES:
        pts = []
        for r in rr:
            i_ratio = surface_brightness(r, 1.0, 1.0, n)     # I/I_e
            mu = brightness_to_mag(i_ratio, mu_e)
            if mu_lo <= mu <= mu_hi:
                pts.append(f"{X(r):.1f},{Y(mu):.1f}")
        parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{col}" stroke-width="2.6"/>')
        parts.append(f'<text x="{x0+90:.1f}" y="{y1+14 + 16*PROFILES.index((label,n,col)):.1f}" '
                     f'fill="{col}" font-size="11">{label}</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
