"""Demo: Gray-Scott reaction-diffusion -- Turing patterns from chemistry.

Prints the autocatalyst growth as a seed develops and an ASCII snapshot of the pattern, then
draws the autocatalyst field at several times, showing structure emerging from a small seed
under the activator-inhibitor dynamics.

    python examples/reaction_diffusion_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from reaction_diffusion import (make_fields, seed_center, evolve, step,  # noqa: E402
                                total_v, pattern_contrast)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Gray-Scott: du/dt = Du lap(u) - u v^2 + F(1-u), dv/dt = Dv lap(v) + u v^2 - (F+k)v\n")
    print("  Seeding autocatalyst in a bare substrate and letting a pattern grow (F=0.035, k=0.06):\n")
    u, v = make_fields(48)
    seed_center(u, v, 8)
    print(f"  {'step':>8}{'total v':>12}{'contrast':>12}")
    for target in (0, 200, 600, 1500):
        while _steps(u) < target:
            u, v = step(u, v, f=0.035, k=0.06)
            _bump(u)
        print(f"  {target:>8}{total_v(v):>12.1f}{pattern_contrast(v):>12.3f}")

    print("\n  ASCII snapshot of the autocatalyst field (denser = more v):")
    _ascii(v)

    print("\n  Two chemicals -- a slow self-promoting activator and a fast inhibitor -- turn a")
    print("  uniform mix unstable, and it settles into standing spots and stripes with no")
    print("  template. Turing's model of morphogenesis: leopard spots, seashell ridges, and more.")

    _svg(os.path.join(outdir, "reaction_diffusion.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'reaction_diffusion.svg')}")


# tiny step counter attached to the field object via a module-level dict (grids are lists)
_counter = {"n": 0}
def _steps(_u):
    return _counter["n"]
def _bump(_u):
    _counter["n"] += 1


def _ascii(v):
    n = len(v)
    shades = " .:-=+*#%@"
    stride = max(1, n // 40)
    for r in range(0, n, stride):
        row = []
        for c in range(0, n, stride):
            idx = min(len(shades) - 1, int(v[r][c] / 0.4 * (len(shades) - 1)))
            row.append(shades[max(0, idx)])
        print("    " + "".join(row))


def _svg(path, size=720, pad=40):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">Gray-Scott reaction-diffusion (Turing patterns)</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'the autocatalyst field growing from a central seed into standing structure</text>',
    ]

    n = 70
    snaps = [(0, "seed"), (600, "t=600"), (1800, "t=1800"), (4000, "t=4000")]
    u, v = make_fields(n)
    seed_center(u, v, 10)
    panel_w = (size - 2 * pad) / 4
    cell = panel_w * 0.92 / n
    top = 60
    done = 0
    for pi, (target, label) in enumerate(snaps):
        while done < target:
            u, v = step(u, v, f=0.035, k=0.06)
            done += 1
        x0 = pad + panel_w * pi + (panel_w - n * cell) / 2
        for r in range(n):
            for c in range(n):
                val = v[r][c]
                if val < 0.05:
                    continue
                t = min(1.0, val / 0.4)
                # blue -> cyan -> yellow ramp with v
                if t < 0.5:
                    u2 = t / 0.5
                    rr = int(0x1b + u2*(0x06-0x1b)); gg = int(0x3a + u2*(0xd6-0x3a)); bb = int(0x8b + u2*(0xf0-0x8b))
                else:
                    u2 = (t-0.5)/0.5
                    rr = int(0x06 + u2*(0xff-0x06)); gg = int(0xd6 + u2*(0xd4-0xd6)); bb = int(0xf0 + u2*(0x3b-0xf0))
                parts.append(f'<rect x="{x0 + c*cell:.1f}" y="{top + r*cell:.1f}" '
                             f'width="{cell+0.4:.1f}" height="{cell+0.4:.1f}" '
                             f'fill="#{rr:02x}{gg:02x}{bb:02x}"/>')
        parts.append(f'<text x="{x0 + n*cell/2:.1f}" y="{top + n*cell + 16:.1f}" fill="#8b949e" '
                     f'font-size="10" text-anchor="middle">{label}</text>')

    parts.append(f'<text x="{pad:.1f}" y="{size-20:.1f}" fill="#8b949e" font-size="11">'
                 f'a small seed self-organizes into spots -- no template, pure reaction + diffusion</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
