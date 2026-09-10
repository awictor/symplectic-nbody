"""Demo: Chandrasekhar dynamical friction -- how satellites sink into galaxies.

Shows the non-monotonic velocity dependence of the drag and how the sinking time
scales inversely with mass, then renders the friction-vs-speed curve to SVG.

    python examples/dynamical_friction_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dynamical_friction import friction_acceleration, sinking_time  # noqa: E402

MSUN = 1.989e30
KPC = 3.086e19
GYR = 3.156e16
VC = 220e3
SIGMA = 150e3
RHO = 1e-21


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Chandrasekhar dynamical friction: satellites spiralling in\n")
    print(f"  sinking time into a Milky-Way-like halo (r0=50 kpc, v_circ=220 km/s):")
    print(f"  {'satellite mass':>18}{'sinking time':>16}")
    print("  " + "-" * 34)
    for m, label in ((1e8, "globular cluster"), (1e10, "LMC-scale"),
                     (1e11, "massive dwarf")):
        t = sinking_time(m * MSUN, 50 * KPC, VC) / GYR
        print(f"  {label + f' ({m:.0e})':>18}{t:>13.2f} Gyr")
    print("\n  Heavier objects sink faster (t ~ 1/M): a massive satellite merges in a")
    print("  few Gyr, while a light globular cluster survives ~a Hubble time. This")
    print("  drags massive black holes to galactic centres and erodes cluster orbits.")

    _svg(os.path.join(outdir, "dynamical_friction.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'dynamical_friction.svg')}")


def _svg(path, size=720, pad=64):
    # friction accel vs speed (in units of sigma), showing the peak
    vs = [0.05 * SIGMA * i for i in range(1, 120)]
    acc = [friction_acceleration(1e10 * MSUN, v, RHO, SIGMA) for v in vs]
    vmax = vs[-1] / SIGMA
    amax = max(acc) * 1.1

    def sx(v_over_sigma):
        return pad + v_over_sigma / vmax * (size - 2 * pad)

    def sy(a):
        return size - pad - a / amax * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    poly = " ".join(f"{sx(vs[i]/SIGMA):.1f},{sy(acc[i]):.1f}" for i in range(len(vs)))
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#8338ec" stroke-width="2.2"/>')
    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Dynamical friction vs speed</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'zero as v-&gt;0 (no wake), peaks near the dispersion, ~1/v^2 at high v</text>')
    parts.append(f'<text x="{pad}" y="{size-pad+24}" fill="#8b949e" font-size="11">'
                 f'speed / velocity dispersion -&gt;</text>')
    parts.append(f'<text x="{pad-10}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'deceleration (m/s^2)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
