"""Demo: cosmic recombination via the Saha equation.

Shows the ionization fraction of hydrogen plunging from 1 to 0 across redshift
z ~ 1400 (T ~ 3700 K) -- when the universe went neutral and released the CMB --
far below the naive kT = 13.6 eV temperature. Renders the recombination curve to
SVG.

    python examples/saha_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from saha import (ionization_at_redshift, recombination_redshift,  # noqa: E402
                  naive_ionization_temperature, T0)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    zrec = recombination_redshift(0.5)
    print("Cosmic recombination: when the universe went neutral\n")
    print(f"  naive guess (kT = 13.6 eV)     : {naive_ionization_temperature():.0f} K")
    print(f"  actual recombination (x = 0.5) : z = {zrec:.0f}, T = {T0*(1+zrec):.0f} K")
    print(f"  ~40x cooler than the naive value, because there are ~1.6 billion")
    print(f"  photons per baryon -- the hot tail keeps hydrogen ionized.\n")
    print(f"  {'redshift':>10}{'temp (K)':>12}{'ionized x':>12}")
    print("  " + "-" * 34)
    for z in (1600, 1400, 1300, 1200, 1100, 1000):
        print(f"  {z:>10}{T0*(1+z):>12.0f}{ionization_at_redshift(z):>12.4f}")
    print("\n  Below z~1100 the universe is neutral and transparent: the photons")
    print("  free-stream to us as the cosmic microwave background.")

    _svg(zrec, os.path.join(outdir, "saha.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'saha.svg')}")


def _svg(zrec, path, size=720, pad=64):
    zs = [1700 - 5 * i for i in range(0, 161)]   # 1700 -> 900
    xs = [ionization_at_redshift(z) for z in zs]
    zmin, zmax = min(zs), max(zs)

    def sx(z):
        return pad + (zmax - z) / (zmax - zmin) * (size - 2 * pad)  # high z on left

    def sy(x):
        return size - pad - x * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    poly = " ".join(f"{sx(zs[i]):.1f},{sy(xs[i]):.1f}" for i in range(len(zs)))
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#ffbe0b" stroke-width="2.2"/>')
    parts.append(f'<line x1="{sx(zrec):.1f}" y1="{pad}" x2="{sx(zrec):.1f}" y2="{size-pad}" '
                 f'stroke="#e63946" stroke-dasharray="4,4"/>')
    parts.append(f'<text x="{sx(zrec)+6:.1f}" y="{pad+16}" fill="#e63946" font-size="12">'
                 f'recombination z~{zrec:.0f}</text>')
    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Cosmic recombination: ionization fraction vs redshift</text>')
    parts.append(f'<text x="{pad}" y="{size-pad+24}" fill="#8b949e" font-size="11">'
                 f'redshift (earlier/hotter to the left) -- the CMB is released as x -&gt; 0</text>')
    parts.append(f'<text x="{pad-10}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'ionized fraction x</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
