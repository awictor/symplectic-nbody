"""Demo: Pell's equation x^2 - D y^2 = 1 solved via the continued fraction of sqrt(D).

Shows the periodic continued fraction of sqrt(D), the fundamental solution (including Fermat's
notorious D = 61), the wild unpredictability of solution size with D, and the recurrence generating
further solutions. Draws the digit-length of the fundamental x against D to show the chaos.

    python examples/pell_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pell import (  # noqa: E402
    cf_sqrt_period, fundamental_solution, negative_pell_solution, solutions, verify, is_square,
)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Pell's equation x^2 - D y^2 = 1, via the continued fraction of sqrt(D)\n")

    print(f"  fundamental solutions (smallest x, y > 0):")
    print(f"    {'D':>4}{'cf period':>18}{'x':>16}{'y':>14}")
    for D in [2, 3, 5, 7, 13, 29, 46, 61]:
        a0, period = cf_sqrt_period(D)
        x, y = fundamental_solution(D)
        cf = f"[{a0};{','.join(map(str, period))}]"
        xs = str(x) if x < 1e12 else f"{x:.3e}"
        print(f"    {D:>4}{cf:>18}{xs:>16}{y:>14}")

    x, y = fundamental_solution(61)
    print(f"\n  Fermat's challenge D=61: x = {x}, y = {y}")
    print(f"  verify: {x}^2 - 61*{y}^2 = {x*x - 61*y*y}")

    # negative Pell
    print(f"\n  negative Pell x^2 - D y^2 = -1 exists iff sqrt(D)'s CF period is odd:")
    for D in [2, 5, 10, 13, 3, 7]:
        neg = negative_pell_solution(D)
        _, period = cf_sqrt_period(D)
        parity = "odd" if len(period) % 2 else "even"
        if neg:
            print(f"    D={D:>2} (period {parity}): ({neg[0]}, {neg[1]}) -> "
                  f"{neg[0]**2 - D*neg[1]**2}")
        else:
            print(f"    D={D:>2} (period {parity}): no solution")

    # solution explosion
    print(f"\n  successive solutions for D=2 grow exponentially:")
    for i, (x, y) in enumerate(solutions(2, 6)):
        print(f"    solution {i+1}: ({x}, {y})")

    # collect digit lengths for the plot
    lengths = []
    for D in range(2, 150):
        if is_square(D):
            continue
        xf, _ = fundamental_solution(D)
        lengths.append((D, len(str(xf))))

    _svg(os.path.join(outdir, "pell.svg"), lengths)
    print(f"\n  The digit-length of the fundamental x jumps chaotically with D -- D=61 needs 10")
    print(f"  digits while its neighbours need 2 or 3. This erratic growth is why Pell is hard.")
    print(f"\n  wrote {os.path.join(outdir, 'pell.svg')}")


def _svg(path, lengths, width=760, height=380):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="15">'
        f'Digit-length of the fundamental solution x of x^2 - D y^2 = 1 vs D (chaotic)</text>',
    ]
    ox, oy, ow, oh = 55, 55, width - 100, height - 100
    dmax = max(d for d, _ in lengths)
    lmax = max(L for _, L in lengths)

    def px(d):
        return ox + ow * d / dmax

    def py(L):
        return oy + oh * (1 - L / lmax)

    parts.append(f'<rect x="{ox}" y="{oy}" width="{ow}" height="{oh}" fill="none" stroke="#30363d"/>')
    for L in range(0, lmax + 1, max(1, lmax // 5)):
        y = py(L)
        parts.append(f'<line x1="{ox}" y1="{y:.1f}" x2="{ox+ow}" y2="{y:.1f}" stroke="#161b22"/>')
        parts.append(f'<text x="{ox-6}" y="{y+3:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="end">{L}</text>')
    for d, L in lengths:
        x = px(d)
        h = oy + oh - py(L)
        color = "#ff6b6b" if L >= 8 else "#4dabf7"
        parts.append(f'<rect x="{x-1.5:.1f}" y="{py(L):.1f}" width="3" height="{h:.1f}" fill="{color}"/>')
        if L >= 8:
            parts.append(f'<text x="{x:.1f}" y="{py(L)-3:.1f}" fill="#ff6b6b" font-size="8" '
                         f'text-anchor="middle">{d}</text>')
    parts.append(f'<text x="{ox+ow/2:.0f}" y="{oy+oh+22:.0f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">D (red = fundamental x has >= 8 digits, e.g. D=61)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
