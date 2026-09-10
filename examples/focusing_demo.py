"""Demo: gravitational focusing and runaway planetary growth.

Shows how the collision cross-section is enhanced over geometry at low encounter
speeds, driving runaway growth of planetesimals, and reverts to geometric at high
speed. Renders the enhancement factor vs encounter speed to a log-log SVG.

    python examples/focusing_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from focusing import (escape_speed, focusing_factor, safronov_number,  # noqa: E402
                      is_runaway)

R = 1e5           # 100 km planetesimal
RHO = 3000.0
M = 4.0 / 3.0 * math.pi * R ** 3 * RHO


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Gravitational focusing (100 km planetesimal)\n")
    print(f"  escape speed: {escape_speed(M, R):.0f} m/s\n")
    print(f"  {'v_inf (m/s)':>12}{'focusing':>12}{'Safronov':>12}{'regime':>12}")
    print("  " + "-" * 48)
    for v in (1, 10, 50, 100, 500, 1000):
        reg = "runaway" if is_runaway(M, R, float(v)) else "geometric"
        print(f"  {v:>12}{focusing_factor(M, R, float(v)):>12.1f}"
              f"{safronov_number(M, R, float(v)):>12.2f}{reg:>12}")
    print("\n  When the swarm is dynamically cold (v_inf << v_esc) gravity bends")
    print("  distant trajectories into collisions, so the biggest bodies grow")
    print("  fastest -- runaway growth that seeds planetary embryos. Stir the swarm")
    print("  up (high v_inf) and only direct hits count: growth slows to geometric.")

    _svg(os.path.join(outdir, "focusing.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'focusing.svg')}")


def _svg(path, size=720, pad=64):
    vs = [10 ** (0 + 0.03 * i) for i in range(0, 101)]  # 1 .. 1000 m/s
    ff = [focusing_factor(M, R, v) for v in vs]
    lv = [math.log10(v) for v in vs]
    lf = [math.log10(f) for f in ff]
    vmin, vmax = lv[0], lv[-1]
    fmin, fmax = min(lf), max(lf)

    def sx(x):
        return pad + (x - vmin) / (vmax - vmin) * (size - 2 * pad)

    def sy(y):
        return size - pad - (y - fmin) / (fmax - fmin) * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    poly = " ".join(f"{sx(lv[i]):.1f},{sy(lf[i]):.1f}" for i in range(len(vs)))
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#2a9d8f" stroke-width="2.2"/>')
    # escape-speed marker (runaway boundary near v_esc)
    lve = math.log10(escape_speed(M, R))
    parts.append(f'<line x1="{sx(lve):.1f}" y1="{pad}" x2="{sx(lve):.1f}" y2="{size-pad}" '
                 f'stroke="#e63946" stroke-dasharray="4,4"/>')
    parts.append(f'<text x="{sx(lve)+6:.1f}" y="{pad+16}" fill="#e63946" font-size="12">'
                 f'v_inf ~ v_esc</text>')
    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Gravitational focusing enhancement vs encounter speed</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'huge (runaway) when v_inf &lt; v_esc, -&gt; 1 (geometric) when fast</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+22}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">log10 encounter speed (m/s) -&gt;</text>')
    parts.append(f'<text x="{pad-10}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'log10 cross-section / geometric</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
