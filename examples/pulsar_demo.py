"""Demo: the Hulse-Taylor binary pulsar -- the first proof of gravitational waves.

Computes PSR B1913+16's orbital-period decay from gravitational-wave emission,
compares it to the measured value, and renders the famous cumulative-periastron-
shift parabola (the plot whose data points fall on the GR curve) to SVG.

    python examples/pulsar_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pulsar import (hulse_taylor_pdot, cumulative_periastron_shift,  # noqa: E402
                    merger_time, HT_P_ORB, HT_E, YEAR)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    pdot = hulse_taylor_pdot()
    print("PSR B1913+16 (Hulse-Taylor): gravitational waves before LIGO\n")
    print(f"  orbital period      : {HT_P_ORB/3600:.3f} hours")
    print(f"  eccentricity        : {HT_E}")
    print(f"  dP/dt (GR predicted): {pdot:.4e} s/s")
    print(f"  dP/dt (measured)    : -2.423e-12 s/s")
    print(f"  agreement           : {pdot / -2.423e-12 * 100:.1f}% of measured\n")
    print(f"  cumulative shift over 30 yr : {cumulative_periastron_shift(30*YEAR):.1f} s")
    print(f"  P/|dP/dt| timescale         : {merger_time()/YEAR/1e6:.0f} Myr\n")
    print("  The orbit's period shrinks by ~76 microseconds per year as the system")
    print("  radiates gravitational waves. Tracking that decay for decades -- the")
    print("  parabola below -- won the 1993 Nobel Prize, 22 years before LIGO.")

    years = [y for y in range(0, 36)]
    shifts = [cumulative_periastron_shift(y * YEAR) for y in years]
    _svg(years, shifts, os.path.join(outdir, "pulsar.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'pulsar.svg')}")


def _svg(years, shifts, path, size=720, pad=64):
    ymax = years[-1]
    smin = min(shifts)

    def sx(y):
        return pad + y / ymax * (size - 2 * pad)

    def sy(s):
        # shifts are negative; 0 at top, smin at bottom
        return pad + (s / smin) * (size - 2 * pad) if smin != 0 else pad

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{size-pad}" y2="{pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    poly = " ".join(f"{sx(years[i]):.1f},{sy(shifts[i]):.1f}" for i in range(len(years)))
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#4cc9f0" stroke-width="2.2"/>')
    # data-like markers every 3 years on the curve
    for i in range(0, len(years), 3):
        parts.append(f'<circle cx="{sx(years[i]):.1f}" cy="{sy(shifts[i]):.1f}" '
                     f'r="3" fill="#ffbe0b"/>')
    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Hulse-Taylor cumulative periastron shift</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'GR prediction (blue) with sample epochs (gold) -- the orbit is decaying</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+22}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">years -&gt;</text>')
    parts.append(f'<text x="{pad-10}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'shift (s), downward</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
