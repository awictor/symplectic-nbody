"""Demo: mean-motion resonance -- a locked resonant argument librates.

Integrates two planets at the 2:1 spacing (resonant) and off it (not), tracks
the 2:1 resonant argument phi in each, and shows phi librating in a bounded range
when locked vs circulating through 2*pi when not. Renders both phi tracks to SVG.

    python examples/resonance_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from resonance import (two_planet_system, resonant_argument_series,  # noqa: E402
                       period_ratio, libration_amplitude)

_BARS = " .:-=+*#@"


def spark(vals, lo=-math.pi, hi=math.pi):
    span = (hi - lo) or 1.0
    return "".join(_BARS[max(0, min(8, int((v - lo) / span * 8)))] for v in vals)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    a_res = 1.0 * 2 ** (2.0 / 3.0)
    print("Mean-motion resonance: 2:1 resonant argument phi\n")

    # resonant
    s = two_planet_system(1.0, a_res, m_inner=2e-3, m_outer=2e-3, e_inner=0.05)
    ts, phis = resonant_argument_series(s, 2, 1, dt=0.002, steps=60000, sample_every=60)
    # off-resonance
    s2 = two_planet_system(1.0, 1.8, m_inner=2e-3, m_outer=2e-3, e_inner=0.05)
    ts2, phis2 = resonant_argument_series(s2, 2, 1, dt=0.002, steps=60000, sample_every=60)

    print(f"  resonant (2:1 spacing) : phi range {libration_amplitude(phis):.2f} rad "
          f"-> LIBRATES (locked)")
    print(f"  off-resonance          : phi range {libration_amplitude(phis2):.2f} rad "
          f"-> circulates (2pi={2*math.pi:.2f})\n")

    stride = max(1, len(phis) // 70)
    print("  resonant phi  :", spark(phis[::stride]))
    print("  off-res  phi  :", spark(phis2[::stride]))
    print("\n  A librating resonant argument IS the lock: the pair's periods stay")
    print("  commensurate. This carves the Kirkwood gaps and builds the Laplace")
    print("  resonance of Io-Europa-Ganymede.")

    _svg(ts, phis, ts2, phis2, os.path.join(outdir, "resonance.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'resonance.svg')}")


def _svg(ts, phis, ts2, phis2, path, size=720, pad=56):
    tmax = max(ts[-1], ts2[-1])

    def sx(t):
        return pad + t / tmax * (size - 2 * pad)

    def sy(phi):  # -pi..pi
        return size / 2 - phi / math.pi * (size / 2 - pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{sy(0):.1f}" x2="{size-pad}" y2="{sy(0):.1f}" '
        f'stroke="#30363d"/>',
    ]
    off = " ".join(f"{sx(ts2[k]):.1f},{sy(phis2[k]):.1f}" for k in range(len(ts2)))
    res = " ".join(f"{sx(ts[k]):.1f},{sy(phis[k]):.1f}" for k in range(len(ts)))
    # off-resonance as faint dots (it wraps, so a polyline would draw jumps)
    parts.append("".join(f'<circle cx="{sx(ts2[k]):.1f}" cy="{sy(phis2[k]):.1f}" '
                         f'r="1" fill="#8b949e" fill-opacity="0.5"/>'
                         for k in range(len(ts2))))
    parts.append("".join(f'<circle cx="{sx(ts[k]):.1f}" cy="{sy(phis[k]):.1f}" '
                         f'r="1.3" fill="#e63946"/>' for k in range(len(ts))))
    parts.append(f'<text x="{pad}" y="30" fill="#e6edf3" font-size="18">'
                 f'2:1 resonant argument: locked (red) vs free (grey)</text>')
    parts.append(f'<text x="{pad}" y="50" fill="#8b949e" font-size="12">'
                 f'phi vs time in [-pi, pi]; red librates about a fixed value, grey fills the range</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
