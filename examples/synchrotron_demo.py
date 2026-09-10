"""Demo: synchrotron radiation -- the radio glow of cosmic magnetic fields.

Shows the critical frequency, single-electron power, and cooling time across
electron energies in a microgauss field, plus the spectral index of a power-law
electron population. Renders critical frequency vs Lorentz factor to a log-log SVG.

    python examples/synchrotron_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from synchrotron import (gyrofrequency, critical_frequency,  # noqa: E402
                         single_electron_power, cooling_time, spectral_index)

YEAR = 3.156e7
B = 1e-9  # 10 microgauss ~ typical ISM/SNR field... use 1 nT here


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print(f"Synchrotron radiation (B = {B*1e9:.0f} nT)\n")
    print(f"  {'gamma':>8}{'nu_c (Hz)':>14}{'P (W)':>12}{'cooling':>14}")
    print("  " + "-" * 48)
    for g in (1e2, 1e3, 1e4, 1e5, 1e6):
        t = cooling_time(g, B)
        tstr = f"{t/YEAR:.1e} yr"
        print(f"  {g:>8.0e}{critical_frequency(g, B):>14.2e}"
              f"{single_electron_power(g, B):>12.2e}{tstr:>14}")
    print("\n  spectral index alpha = (p-1)/2 of a power-law electron population:")
    for p in (2.0, 2.5, 3.0):
        print(f"    p={p}: alpha={spectral_index(p):.2f}  (S(nu) ~ nu^-{spectral_index(p):.2f})")
    print("\n  Relativistic electrons in weak cosmic fields shine in the radio; the")
    print("  spectral slope reveals the electron energy distribution -- how we read")
    print("  jets, radio galaxies, and supernova remnants.")

    _svg(os.path.join(outdir, "synchrotron.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'synchrotron.svg')}")


def _svg(path, size=720, pad=64):
    gs = [10 ** (2 + 0.04 * i) for i in range(0, 101)]  # 1e2 .. 1e6
    nu = [critical_frequency(g, B) for g in gs]
    lg = [math.log10(g) for g in gs]
    ln = [math.log10(n) for n in nu]
    gmin, gmax = lg[0], lg[-1]
    nmin, nmax = ln[0], ln[-1]

    def sx(x):
        return pad + (x - gmin) / (gmax - gmin) * (size - 2 * pad)

    def sy(y):
        return size - pad - (y - nmin) / (nmax - nmin) * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    poly = " ".join(f"{sx(lg[i]):.1f},{sy(ln[i]):.1f}" for i in range(len(gs)))
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#4cc9f0" stroke-width="2.2"/>')
    # radio band marker (1e8 - 1e11 Hz)
    for f, label in ((1e9, "GHz (radio)"),):
        if nmin < math.log10(f) < nmax:
            parts.append(f'<line x1="{pad}" y1="{sy(math.log10(f)):.1f}" x2="{size-pad}" '
                         f'y2="{sy(math.log10(f)):.1f}" stroke="#8b949e" stroke-dasharray="3,4"/>')
            parts.append(f'<text x="{size-pad-4}" y="{sy(math.log10(f))-6:.1f}" fill="#8b949e" '
                         f'font-size="11" text-anchor="end">{label}</text>')
    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Synchrotron critical frequency vs electron energy</text>')
    parts.append(f'<text x="{pad}" y="{size-pad+24}" fill="#8b949e" font-size="11">'
                 f'log10 Lorentz factor gamma; nu_c ~ gamma^2 B</text>')
    parts.append(f'<text x="{pad-10}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'log10 nu_c (Hz)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
