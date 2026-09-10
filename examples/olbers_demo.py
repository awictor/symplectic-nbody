"""Demo: Olbers' paradox -- why the night sky is dark.

Prints the mean free path to a star, the horizon distance, and the fraction of sky
covered at various distances, then draws the covering fraction vs distance with the
cosmic horizon marked far short of where stars would tile the sky.

    python examples/olbers_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from olbers import (mean_free_path_to_star, covering_fraction,  # noqa: E402
                    horizon_distance, sky_fraction_within_horizon, LY, PC)

N_STARS = 0.1 / PC ** 3


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    mfp = mean_free_path_to_star(N_STARS)
    hor = horizon_distance()
    print("Olbers' paradox: in an infinite static universe the sky would blaze\n")
    print(f"  star density ~ 0.1 /pc^3  ->  mean free path to a star = "
          f"{mfp/LY:.1e} ly")
    print(f"  cosmic horizon (c x age)  = {hor/LY:.1e} ly")
    print(f"  ratio horizon / mfp       = {hor/mfp:.1e}\n")
    print(f"  {'distance':>16}{'d / mfp':>12}{'sky covered':>16}")
    print("  " + "-" * 44)
    for label, d in [("cosmic horizon", hor), ("100x horizon", 100 * hor),
                     ("0.1 mfp", 0.1 * mfp), ("1 mfp", mfp),
                     ("5 mfp", 5 * mfp), ("20 mfp", 20 * mfp)]:
        print(f"  {label:>16}{d/mfp:>12.2e}{covering_fraction(d, mfp):>16.4g}")

    print(f"\n  Within the observable universe only ~{sky_fraction_within_horizon(N_STARS):.0e}")
    print("  of the sky is covered by stellar disks -- which is why night is dark. The")
    print("  paradox is real: every line of sight WOULD end on a star, but only after")
    print("  ~10^16 light-years, a million times farther than light has travelled since")
    print("  the Big Bang. The finite age of the universe, not infinite space, saves us.")

    _svg(os.path.join(outdir, "olbers.svg"), mfp, hor)
    print(f"\n  wrote {os.path.join(outdir, 'olbers.svg')}")


def _svg(path, mfp, hor, size=720, pad=72):
    # x = log10(distance / mfp), y = covering fraction (linear 0..1)
    xs = [-7 + 0.1 * i for i in range(0, 91)]   # 1e-7 .. ~1e2 in units of mfp
    fr = [covering_fraction(10 ** x * mfp, mfp) for x in xs]
    xmin, xmax = xs[0], xs[-1]

    def sx(x):
        return pad + (x - xmin) / (xmax - xmin) * (size - 2 * pad)

    def sy(y):
        return size - pad - y * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    # cosmic-horizon vertical line (in units of mfp)
    xh = math.log10(hor / mfp)
    if xmin <= xh <= xmax:
        hx = sx(xh)
        parts.append(f'<line x1="{hx:.1f}" y1="{pad}" x2="{hx:.1f}" y2="{size-pad}" '
                     f'stroke="#ff6b6b" stroke-width="1.4" stroke-dasharray="5 4"/>')
        parts.append(f'<text x="{hx+6:.1f}" y="{pad+40:.1f}" fill="#ff6b6b" '
                     f'font-size="11">cosmic horizon (we see only this far)</text>')
    # region beyond horizon shaded as "the blazing sky we cannot reach"
    poly = " ".join(f"{sx(xs[i]):.1f},{sy(fr[i]):.1f}" for i in range(len(xs)))
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#8338ec" stroke-width="2.6"/>')

    # dot at the horizon covering fraction
    if xmin <= xh <= xmax:
        parts.append(f'<circle cx="{sx(xh):.1f}" cy="{sy(covering_fraction(hor, mfp)):.1f}" '
                     f'r="5" fill="#ffd43b"/>')
        parts.append(f'<text x="{sx(xh)-6:.1f}" y="{size-pad-8:.1f}" fill="#ffd43b" '
                     f'font-size="10" text-anchor="end">sky ~0 covered -> dark</text>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Olbers: sky covered by stars vs distance</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'stars would tile the sky near 1 mfp -- but the horizon is 10^-6 of that</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+24}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">log10 (distance / mean free path) -&gt;</text>')
    parts.append(f'<text x="{pad-18}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'fraction of sky covered by stars</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
