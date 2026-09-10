"""Demo: adiabatic compression and expansion.

Prints the temperature reached by compressing air at various ratios (diesel ignition)
and the corrected speed of sound, then draws an adiabat and an isotherm through the
same point on a P-V diagram -- the adiabat is steeper because temperature changes too.

    python examples/adiabatic_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from adiabatic import (pressure_after, temperature_after_volume,  # noqa: E402
                       adiabatic_work, sound_speed, GAMMA_DIATOMIC)

M_AIR = 0.02896


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Adiabatic: P V^gamma = const, T V^(gamma-1) = const (no heat flow)\n")
    print(f"  speed of sound in air (293 K): {sound_speed(293, M_AIR):.1f} m/s")
    print(f"    (Laplace's gamma vs Newton's wrong {sound_speed(293, M_AIR, 1.0):.0f} m/s)\n")
    print(f"  adiabatic compression of air from 300 K:")
    print(f"  {'ratio V1/V2':>14}{'T2 (K)':>10}{'P2/P1':>10}")
    print("  " + "-" * 34)
    for r in (2, 5, 10, 22, 50):
        T2 = temperature_after_volume(300.0, float(r), 1.0)
        P2 = pressure_after(1.0, float(r), 1.0)
        print(f"  {r:>14}{T2:>10.0f}{P2:>10.1f}")

    print("\n  Compressing a gas with no time to shed heat drives its temperature up")
    print("  steeply -- a diesel engine's 22:1 squeeze reaches ~1000 K and ignites fuel")
    print("  without a spark. Expansion does the reverse, cooling the gas (rising air,")
    print("  released CO2). Sound waves compress air adiabatically too, so the speed of")
    print("  sound carries Laplace's sqrt(gamma) factor -- the fix to Newton's estimate.")

    _svg(os.path.join(outdir, "adiabatic.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'adiabatic.svg')}")


def _svg(path, size=720, pad=72):
    # P-V diagram: adiabat P = P0 (V0/V)^gamma vs isotherm P = P0 V0/V
    P0, V0 = 1.0, 1.0
    Vs = [0.4 + 0.02 * i for i in range(0, 131)]   # 0.4 .. 3.0
    adiab = [P0 * (V0 / V) ** GAMMA_DIATOMIC for V in Vs]
    iso = [P0 * V0 / V for V in Vs]
    xmin, xmax = Vs[0], Vs[-1]
    ymax = max(max(adiab), max(iso)) * 0.6   # clip the steep low-V spike for display
    ymin = 0.0

    def sx(x):
        return pad + (x - xmin) / (xmax - xmin) * (size - 2 * pad)

    def sy(y):
        return size - pad - (min(y, ymax) - ymin) / (ymax - ymin) * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    pa = " ".join(f"{sx(Vs[i]):.1f},{sy(adiab[i]):.1f}" for i in range(len(Vs)))
    pi = " ".join(f"{sx(Vs[i]):.1f},{sy(iso[i]):.1f}" for i in range(len(Vs)))
    parts.append(f'<polyline points="{pi}" fill="none" stroke="#4dabf7" stroke-width="2.2"/>')
    parts.append(f'<polyline points="{pa}" fill="none" stroke="#ff922b" stroke-width="2.6"/>')

    # common point at V0
    parts.append(f'<circle cx="{sx(1.0):.1f}" cy="{sy(1.0):.1f}" r="4" fill="#ffd43b"/>')

    parts.append(f'<text x="{pad+10}" y="{pad+22}" fill="#ff922b" font-size="12">'
                 f'adiabat P V^1.4 (steeper -- T changes)</text>')
    parts.append(f'<text x="{pad+10}" y="{pad+38}" fill="#4dabf7" font-size="12">'
                 f'isotherm P V (T constant)</text>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Adiabat vs isotherm on a P-V diagram</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'the adiabat is steeper: compressing without heat loss also raises T</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+24}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">volume (rel.) -&gt;</text>')
    parts.append(f'<text x="{pad-18}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'pressure (rel.)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
