"""Demo: Fermi acceleration and the universal cosmic-ray spectrum.

Prints the compression ratio and spectral index vs shock strength, then draws the
resulting power-law spectra N(E) ~ E^(-p) for a range of shock Mach numbers, with
the strong-shock E^(-2) line highlighted.

    python examples/fermi_acceleration_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fermi_acceleration import (compression_ratio, spectral_index_from_mach,  # noqa: E402
                                energy_gain_per_cycle, power_law, strong_shock_index)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Fermi acceleration: shocks stamp a power law N(E) ~ E^(-p)\n")
    print(f"  strong-shock (M -> inf) index: p = {strong_shock_index():.2f} "
          f"-> the universal E^(-2) spectrum\n")
    print(f"  {'Mach':>7}{'compression r':>15}{'index p':>10}")
    print("  " + "-" * 32)
    for M in (1.5, 2, 3, 5, 10, 50, 1000):
        r = compression_ratio(M)
        p = spectral_index_from_mach(M)
        print(f"  {M:>7g}{r:>15.3f}{p:>10.3f}")

    b = 0.03
    print(f"\n  per-cycle energy gain at beta = {b}:")
    print(f"    first-order (shock): {energy_gain_per_cycle(b, 1):.4f}  (~4/3 beta)")
    print(f"    second-order (clouds): {energy_gain_per_cycle(b, 2):.5f}  (~4/3 beta^2)")
    print(f"    first-order wins by 1/beta ~ {1/b:.0f}x -- why shocks dominate.")

    print("\n  The magic of diffusive shock acceleration: the index depends only on")
    print("  the compression ratio, p = (r+2)/(r-1), not on the messy microphysics.")
    print("  Every strong shock converges to r=4, p=2 -- so supernova remnants across")
    print("  the Galaxy all inject nearly the same E^(-2) cosmic-ray spectrum.")

    _svg(os.path.join(outdir, "fermi_acceleration.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'fermi_acceleration.svg')}")


def _svg(path, size=720, pad=68):
    machs = [(1.5, "#ff6b6b", "M=1.5"), (2.0, "#ff922b", "M=2"),
             (3.0, "#ffd43b", "M=3"), (5.0, "#06d6a0", "M=5"),
             (1000.0, "#4dabf7", "M>>1 (p=2)")]
    E = [10 ** (0.1 * i) for i in range(0, 61)]   # E/E0 from 1 to 1e6
    lx = [math.log10(x) for x in E]
    xmin, xmax = lx[0], lx[-1]
    # y-range: use the steepest (M=1.5) at top E for the floor
    p_steep = spectral_index_from_mach(1.5)
    ymin = math.log10(power_law(E[-1], 1.0, p_steep))
    ymax = 0.0

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
    ytop = pad + 22
    for i, (M, col, label) in enumerate(machs):
        p = spectral_index_from_mach(M)
        ys = [math.log10(power_law(x, 1.0, p)) for x in E]
        width = 3.0 if M >= 1000 else 2.0
        poly = " ".join(f"{sx(lx[j]):.1f},{sy(ys[j]):.1f}" for j in range(len(E)))
        parts.append(f'<polyline points="{poly}" fill="none" stroke="{col}" stroke-width="{width}"/>')
        parts.append(f'<text x="{size-pad-120}" y="{ytop + i*16}" fill="{col}" '
                     f'font-size="12">{label} (p={p:.2f})</text>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Diffusive shock acceleration: N(E) ~ E^(-p)</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'p = (r+2)/(r-1); every strong shock converges to r=4, p=2</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+24}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">log10 energy (E / E0) -&gt;</text>')
    parts.append(f'<text x="{pad-14}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'log10 N(E)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
