"""Demo: Big Bang nucleosynthesis -- the primordial 25% helium.

Traces the neutron-to-proton ratio from equilibrium through weak freeze-out and
neutron decay to the ~0.25 helium mass fraction, and renders n/p vs temperature
to SVG.

    python examples/bbn_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bbn import (np_ratio_equilibrium, np_ratio_at_freezeout,  # noqa: E402
                 np_ratio_after_decay, helium_mass_fraction, T_FREEZE_MEV)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Big Bang nucleosynthesis: the origin of primordial helium\n")
    print(f"  {'stage':<34}{'n/p':>8}")
    print("  " + "-" * 42)
    print(f"  {'equilibrium at 10 MeV (t~0.01 s)':<34}{np_ratio_equilibrium(10.0):>8.3f}")
    print(f"  {'freeze-out at 0.8 MeV (t~1 s)':<34}{np_ratio_at_freezeout():>8.3f}")
    print(f"  {'after neutron decay (t~200 s)':<34}{np_ratio_after_decay():>8.3f}")
    print(f"\n  primordial helium mass fraction Y_p = {helium_mass_fraction():.3f}")
    print(f"  (observed ~0.245-0.25 -- a triumph of the hot Big Bang)\n")
    print("  Neutrons and protons start nearly equal, the ratio freezes at ~1/6")
    print("  as the weak interaction shuts off, decays to ~1/7, and nearly all")
    print("  surviving neutrons end up in helium-4: about a quarter of all mass.")

    _svg(os.path.join(outdir, "bbn.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'bbn.svg')}")


def _svg(path, size=720, pad=64):
    # n/p equilibrium vs temperature (log-T axis, high T on left)
    Ts = [10 ** (1.3 - 0.02 * i) for i in range(0, 110)]  # ~20 MeV -> ~0.1 MeV
    rs = [np_ratio_equilibrium(T) for T in Ts]
    lT = [math.log10(T) for T in Ts]
    Tmin, Tmax = min(lT), max(lT)

    def sx(x):
        return pad + (Tmax - x) / (Tmax - Tmin) * (size - 2 * pad)  # hot on left

    def sy(r):
        return size - pad - r * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    poly = " ".join(f"{sx(lT[i]):.1f},{sy(rs[i]):.1f}" for i in range(len(Ts)))
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#4cc9f0" stroke-width="2.2"/>')
    # freeze-out line
    lf = math.log10(T_FREEZE_MEV)
    parts.append(f'<line x1="{sx(lf):.1f}" y1="{pad}" x2="{sx(lf):.1f}" y2="{size-pad}" '
                 f'stroke="#e63946" stroke-dasharray="4,4"/>')
    parts.append(f'<text x="{sx(lf)+6:.1f}" y="{pad+16}" fill="#e63946" font-size="12">'
                 f'freeze-out (0.8 MeV, n/p~1/6)</text>')
    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Neutron/proton ratio vs temperature</text>')
    parts.append(f'<text x="{pad}" y="{size-pad+24}" fill="#8b949e" font-size="11">'
                 f'log10 temperature (MeV), hot to the left -- freezing sets the helium yield</text>')
    parts.append(f'<text x="{pad-10}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'n/p ratio</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
