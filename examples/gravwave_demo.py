"""Demo: the gravitational-wave chirp of an inspiralling binary.

Adds the 2.5PN radiation-reaction force to a circular binary. The orbit loses
energy to gravitational waves, shrinks, and its frequency sweeps upward -- the
"chirp" LIGO detects. Validates the energy-loss rate against Peters (1964) and
renders the shrinking spiral + the rising frequency track to SVG.

    python examples/gravwave_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gravwave import inspiral, coalescence_time, _rk4_rel  # noqa: E402

_BARS = " .:-=+*#@"
G, M, mu = 1.0, 1.0, 0.25
m1 = m2 = 0.5


def sparkline(vals):
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    return "".join(_BARS[min(8, int((v - lo) / span * 8))] for v in vals)


def peters_dedt(a, c):
    return -(32.0 / 5.0) * mu * mu * M ** 3 / (c ** 5 * a ** 5)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    a0, period = 1.0, 2.0 * math.pi
    # c chosen so the binary merges in ~40 orbits (GR amplified to be visible)
    c = (40 * period * 256 * mu * M * M / (5 * a0 ** 4)) ** 0.2
    print(f"Gravitational-wave inspiral (equal-mass binary, GR amplified: v/c~{1/c:.2f})\n")

    # 1. validate energy-loss rate against Peters (1964)
    r, v = [a0, 0.0, 0.0], [0.0, math.sqrt(M / a0), 0.0]
    dt = period / 3000
    E0 = 0.5 * mu * v[1] ** 2 - mu * M / a0
    for _ in range(3000 * 8):
        r, v = _rk4_rel(r, v, m1, m2, c, dt)
    d = math.sqrt(sum(x * x for x in r))
    E1 = 0.5 * mu * sum(x * x for x in v) - mu * M / d
    measured = (E1 - E0) / (period * 8)
    print(f"  energy-loss rate  dE/dt: measured {measured:.3e}  "
          f"Peters {peters_dedt(1.0, c):.3e}  ratio {measured/peters_dedt(1.0,c):.3f}")

    # 2. full inspiral
    tc = coalescence_time(a0, 0.0, m1, m2, c)
    ts, seps, freqs = inspiral(a0, m1, m2, c, dt=period / 3000,
                               n_steps=int(tc / (period / 3000) * 0.98),
                               sample_every=150)
    print(f"\n  separation: {seps[0]:.2f} -> {seps[-1]:.2f}  "
          f"(orbit shrinks as it radiates)")
    print(f"  orbital frequency chirps up {freqs[-1]/freqs[0]:.1f}x:")
    stride = max(1, len(freqs) // 70)
    print("  " + sparkline(freqs[::stride]))

    # 3. render the inspiral spiral by re-integrating and recording xy
    r, v = [a0, 0.0, 0.0], [0.0, math.sqrt(M / a0), 0.0]
    pts = [(a0, 0.0)]
    n = int(tc / dt * 0.98)
    for i in range(n):
        r, v = _rk4_rel(r, v, m1, m2, c, dt)
        if i % 40 == 0:
            pts.append((r[0], r[1]))
    _spiral_svg(pts, os.path.join(outdir, "gw_inspiral.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'gw_inspiral.svg')}")
    print("  Same mechanism as GW150914: the binary radiates orbital energy as")
    print("  gravitational waves and spirals to merger, chirping as it goes.")


def _spiral_svg(pts, path, size=640):
    ext = max(max(abs(x), abs(y)) for x, y in pts) * 1.1

    def sx(x): return size / 2 + x / ext * (size / 2 - 20)
    def sy(y): return size / 2 - y / ext * (size / 2 - 20)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#05070d"/>',
    ]
    poly = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in pts)
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#4cc9f0" '
                 f'stroke-width="1.2" stroke-opacity="0.9"/>')
    parts.append(f'<circle cx="{sx(0):.1f}" cy="{sy(0):.1f}" r="4" fill="#ffd166"/>')
    parts.append(f'<circle cx="{sx(pts[-1][0]):.1f}" cy="{sy(pts[-1][1]):.1f}" '
                 f'r="3.5" fill="#ff70a6"/>')
    parts.append(f'<text x="16" y="26" fill="#e6edf3" font-size="15">'
                 f'Gravitational-wave inspiral</text>')
    parts.append(f'<text x="16" y="{size-14}" fill="#8b949e" font-size="11">'
                 f'relative orbit shrinks to merger as it radiates (GR amplified)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
