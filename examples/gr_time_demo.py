"""Demo: gravitational time effects -- redshift, GPS, and the Shapiro delay.

Reproduces the Pound-Rebka redshift, the GPS relativistic clock correction, the
Sun's surface redshift, and the Shapiro radar delay, then renders the Shapiro
delay vs impact parameter (diverging as the ray grazes the Sun) to SVG.

    python examples/gr_time_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gr_time import (redshift_uniform_field, gps_time_gain_per_day,  # noqa: E402
                     gravitational_redshift, shapiro_delay,
                     M_SUN, R_SUN, AU)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Gravitational time effects (classic GR tests)\n")
    print(f"  Pound-Rebka (22.5 m tower)  : z = {redshift_uniform_field(9.8, 22.5):.2e}"
          f"   (measured 2.5e-15)")
    print(f"  GPS clock gain              : {gps_time_gain_per_day()*1e6:+.1f} us/day"
          f"  (must be corrected or GPS drifts km/day)")
    print(f"  Sun surface redshift        : z = {gravitational_redshift(M_SUN, R_SUN, 1e15):.2e}"
          f"   (2.12e-6)")
    print(f"  Shapiro delay (past the Sun): {shapiro_delay(AU, 8.5*AU, R_SUN)*1e6:.0f} us"
          f"      (Cassini ~240-280 us)\n")
    print("  Clocks run slower deeper in gravity and light lags near mass. The GPS")
    print("  correction is relativity you rely on daily; the Shapiro delay is the")
    print("  tightest Solar-System test of general relativity.")

    _svg(os.path.join(outdir, "gr_time.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'gr_time.svg')}")


def _svg(path, size=720, pad=64):
    # Shapiro delay vs impact parameter b (in solar radii), grazing -> larger
    bs = [R_SUN * (1.0 + 0.1 * i) for i in range(0, 200)]
    dts = [shapiro_delay(AU, 8.5 * AU, b) * 1e6 for b in bs]  # microseconds
    bmin, bmax = bs[0] / R_SUN, bs[-1] / R_SUN
    dmax = max(dts)
    dmin = min(dts)

    def sx(bsun):
        return pad + (bsun - bmin) / (bmax - bmin) * (size - 2 * pad)

    def sy(dt):
        return size - pad - (dt - dmin) / (dmax - dmin) * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    poly = " ".join(f"{sx(bs[i]/R_SUN):.1f},{sy(dts[i]):.1f}" for i in range(len(bs)))
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#4cc9f0" stroke-width="2"/>')
    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Shapiro delay vs impact parameter</text>')
    parts.append(f'<text x="{pad}" y="{size-pad+24}" fill="#8b949e" font-size="11">'
                 f'impact parameter (solar radii); grazing rays are delayed most</text>')
    parts.append(f'<text x="{pad-10}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'round-trip delay (microseconds)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
