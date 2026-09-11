"""Demo: the birthday problem -- coincidences are more common than they feel.

Prints the collision probability at the famous group sizes against a Monte-Carlo run, then
draws the collision-probability curve versus group size (with 23 marked where it crosses 50%)
and how the median crossover grows like sqrt(days) across day-counts.

    python examples/birthday_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from birthday import (prob_collision, collision_approx, min_people_for,  # noqa: E402
                      median_collision, expected_first_collision, simulate_collision)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Birthday problem: with 365 days, just 23 people make a shared birthday likelier than not\n")
    print(f"  {'people k':>9}{'exact':>10}{'Poisson':>10}{'sim':>9}")
    for k in (10, 23, 40, 57, 70):
        print(f"  {k:>9}{prob_collision(k):>10.4f}{collision_approx(k):>10.4f}"
              f"{simulate_collision(k, trials=6000, seed=5):>9.4f}")

    print(f"\n  50% at k = {min_people_for(0.5)},  99% at k = {min_people_for(0.99)},  "
          f"99.9% at k = {min_people_for(0.999)}.")
    print(f"  The crossover grows only like ~1.177 sqrt(days) (~{median_collision():.1f}); the first")
    print(f"  collision arrives after ~{expected_first_collision():.0f} draws. It is the number of")
    print("  PAIRS, ~k^2/2, that drives collisions -- the same square-root law makes a b-bit hash")
    print("  collide after ~2^(b/2) tries (the birthday attack), not 2^b.")

    _svg(os.path.join(outdir, "birthday.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'birthday.svg')}")


def _svg(path, w=760, h=380):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="monospace">',
        f'<rect width="{w}" height="{h}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'The birthday problem: 23 people, better-than-even odds</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'collision probability vs group size for 365 days (left); '
        f'the 50% crossover ~ sqrt(days) (right)</text>',
    ]

    # left: P(collision) vs k for 365 days, exact curve + Poisson approx
    days = 365
    lx0, lx1 = 55, w // 2 - 20
    ly0, ly1 = h - 50, 60
    kmax = 80

    def LX(k):
        return lx0 + k / kmax * (lx1 - lx0)

    def LY(p):
        return ly0 - p * (ly0 - ly1)

    parts.append(f'<line x1="{lx0}" y1="{ly0}" x2="{lx1}" y2="{ly0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{lx0}" y1="{ly0}" x2="{lx0}" y2="{ly1}" stroke="#8b949e" stroke-width="1.2"/>')
    # 50% gridline
    parts.append(f'<line x1="{lx0}" y1="{LY(0.5):.1f}" x2="{lx1}" y2="{LY(0.5):.1f}" '
                 f'stroke="#21262d" stroke-width="1"/>')
    exact = " ".join(f"{LX(k):.1f},{LY(prob_collision(k)):.1f}" for k in range(1, kmax + 1))
    approxc = " ".join(f"{LX(k):.1f},{LY(collision_approx(k)):.1f}" for k in range(1, kmax + 1))
    parts.append(f'<polyline points="{approxc}" fill="none" stroke="#ff922b" stroke-width="1.4" '
                 f'stroke-dasharray="4 3"/>')
    parts.append(f'<polyline points="{exact}" fill="none" stroke="#4dabf7" stroke-width="2.5"/>')
    # mark k = 23
    k50 = min_people_for(0.5)
    parts.append(f'<line x1="{LX(k50):.1f}" y1="{ly1}" x2="{LX(k50):.1f}" y2="{ly0}" '
                 f'stroke="#ff6b6b" stroke-width="1" stroke-dasharray="3 3"/>')
    parts.append(f'<circle cx="{LX(k50):.1f}" cy="{LY(prob_collision(k50)):.1f}" r="3.5" fill="#ff6b6b"/>')
    parts.append(f'<text x="{LX(k50)+5:.1f}" y="{LY(0.5)-6:.1f}" fill="#ff6b6b" font-size="10">'
                 f'k={k50}, 50%</text>')
    for p in (0.0, 0.5, 1.0):
        parts.append(f'<text x="{lx0-6:.1f}" y="{LY(p)+3:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="end">{p:.1f}</text>')
    for k in (0, 40, 80):
        parts.append(f'<text x="{LX(k):.1f}" y="{ly0+16:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="middle">{k}</text>')
    parts.append(f'<text x="{(lx0+lx1)/2:.1f}" y="{ly0+30:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">group size k</text>')
    # legend
    parts.append(f'<rect x="{lx0+8}" y="{ly1}" width="9" height="9" fill="#4dabf7"/>'
                 f'<text x="{lx0+21}" y="{ly1+8}" fill="#e6edf3" font-size="9">exact</text>')
    parts.append(f'<rect x="{lx0+8}" y="{ly1+14}" width="9" height="9" fill="#ff922b"/>'
                 f'<text x="{lx0+21}" y="{ly1+22}" fill="#e6edf3" font-size="9">Poisson approx</text>')

    # right: 50% crossover k* vs number of days -- the square-root law
    rx0, rx1 = w // 2 + 45, w - 30
    ry0, ry1 = h - 50, 60
    day_list = [23, 50, 100, 200, 365, 500, 1000, 2000, 4000]
    kstars = [min_people_for(0.5, days=d) for d in day_list]
    dmax, kmx = day_list[-1], max(kstars) * 1.1

    def RX(d):
        return rx0 + d / dmax * (rx1 - rx0)

    def RY(k):
        return ry0 - k / kmx * (ry0 - ry1)

    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx1}" y2="{ry0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx0}" y2="{ry1}" stroke="#8b949e" stroke-width="1.2"/>')
    # sqrt reference curve 1.177 sqrt(d)
    sqrt_pts = " ".join(f"{RX(d):.1f},{RY(1.1774*math.sqrt(d)):.1f}"
                        for d in range(23, dmax + 1, 40))
    parts.append(f'<polyline points="{sqrt_pts}" fill="none" stroke="#8b949e" stroke-width="1" '
                 f'stroke-dasharray="2 3"/>')
    dots = " ".join(f"{RX(d):.1f},{RY(k):.1f}" for d, k in zip(day_list, kstars))
    parts.append(f'<polyline points="{dots}" fill="none" stroke="#06d6a0" stroke-width="2"/>')
    for d, k in zip(day_list, kstars):
        parts.append(f'<circle cx="{RX(d):.1f}" cy="{RY(k):.1f}" r="2.5" fill="#06d6a0"/>')
    # mark 365
    parts.append(f'<text x="{RX(365):.1f}" y="{RY(23)-8:.1f}" fill="#06d6a0" font-size="9" '
                 f'text-anchor="middle">365 -> 23</text>')
    parts.append(f'<text x="{(rx0+rx1)/2:.1f}" y="{ry0+16:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">number of days d</text>')
    parts.append(f'<text x="{rx0-6:.1f}" y="{ry1-2:.1f}" fill="#8b949e" font-size="9" '
                 f'text-anchor="end">k* (50%)</text>')
    parts.append(f'<text x="{rx1-2:.1f}" y="{RY(1.1774*math.sqrt(dmax))-4:.1f}" fill="#8b949e" '
                 f'font-size="9" text-anchor="end">1.177 sqrt(d)</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
