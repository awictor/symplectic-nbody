"""Demo: the Larmor formula -- radiation from accelerating charges.

Shows the quadratic power law, the relativistic gamma^4 (circular) and gamma^6
(linear) boosts, and the classical-atom collapse time, then renders the radiated
power vs Lorentz factor to a log-log SVG.

    python examples/larmor_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from larmor import (larmor_power, relativistic_power_perpendicular,  # noqa: E402
                    relativistic_power_parallel, atom_collapse_time)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    a = 1e20  # m/s^2 (huge, e.g. in a strong field)
    print("The Larmor formula: accelerating charges radiate (P ~ a^2)\n")
    print(f"  non-relativistic power at a={a:.0e} m/s^2: {larmor_power(a):.2e} W\n")
    print(f"  {'gamma':>8}{'perp (gamma^4)':>18}{'parallel (gamma^6)':>20}")
    print("  " + "-" * 46)
    for g in (1, 10, 100, 1000):
        print(f"  {g:>8}{relativistic_power_perpendicular(a, float(g)):>18.2e}"
              f"{relativistic_power_parallel(a, float(g)):>20.2e}")
    print(f"\n  classical hydrogen atom collapse time: {atom_collapse_time():.2e} s")
    print("  -- an orbiting electron radiating by Larmor spirals in almost instantly.")
    print("  That catastrophe is what quantum mechanics had to fix. The same formula,")
    print("  boosted by gamma^4, is the engine of synchrotron radiation.")

    _svg(a, os.path.join(outdir, "larmor.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'larmor.svg')}")


def _svg(a, path, size=720, pad=64):
    gammas = [10 ** (0.03 * i) for i in range(0, 101)]  # 1 .. 1000
    perp = [relativistic_power_perpendicular(a, g) for g in gammas]
    par = [relativistic_power_parallel(a, g) for g in gammas]
    lg = [math.log10(g) for g in gammas]
    lp = [math.log10(p) for p in perp]
    lpar = [math.log10(p) for p in par]
    gmin, gmax = lg[0], lg[-1]
    ymin = min(min(lp), min(lpar))
    ymax = max(max(lp), max(lpar))

    def sx(x):
        return pad + (x - gmin) / (gmax - gmin) * (size - 2 * pad)

    def sy(y):
        return size - pad - (y - ymin) / (ymax - ymin) * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    pp = " ".join(f"{sx(lg[i]):.1f},{sy(lp[i]):.1f}" for i in range(len(gammas)))
    parp = " ".join(f"{sx(lg[i]):.1f},{sy(lpar[i]):.1f}" for i in range(len(gammas)))
    parts.append(f'<polyline points="{pp}" fill="none" stroke="#4cc9f0" stroke-width="2"/>')
    parts.append(f'<polyline points="{parp}" fill="none" stroke="#ff006e" stroke-width="2"/>')
    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Radiated power vs Lorentz factor (log-log)</text>')
    parts.append(f'<text x="{pad+14}" y="{pad+8}" fill="#ff006e" font-size="12">'
                 f'parallel accel ~ gamma^6 (linear accelerator)</text>')
    parts.append(f'<text x="{pad+14}" y="{pad+26}" fill="#4cc9f0" font-size="12">'
                 f'perpendicular accel ~ gamma^4 (synchrotron)</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+22}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">log10 gamma -&gt;</text>')
    parts.append(f'<text x="{pad-10}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'log10 P (W)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
