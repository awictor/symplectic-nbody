"""Demo: the coupon collector -- how long to collect the whole set.

Prints the expected collection time E[T] = n H_n against a Monte-Carlo mean for several set
sizes, then draws the collection-progress curve (distinct coupons held versus draws, showing
how the last few coupons dominate the wait) and the completion-probability CDF.

    python examples/coupon_collector_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from coupon_collector import (expected_time, expected_time_approx, std_dev,  # noqa: E402
                              expected_partial, prob_complete_by, simulate)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Coupon collector: expected draws to collect all n coupons is E[T] = n * H_n\n")
    print(f"  {'n':>5}{'E[T]':>10}{'n ln n+..':>12}{'std dev':>10}{'sim mean':>10}")
    for n in (6, 20, 50, 100):
        m, _ = simulate(n, trials=3000, seed=7)
        print(f"  {n:>5}{expected_time(n):>10.2f}{expected_time_approx(n):>12.2f}"
              f"{std_dev(n):>10.2f}{m:>10.2f}")

    n = 50
    print(f"\n  For n = {n}: the first half of the coupons costs only "
          f"{expected_partial(n, n//2):.0f} draws, but the last one alone")
    print(f"  averages another {n} -- the tail dominates. Getting from 49 to 50 takes as long")
    print("  as getting the first 35. That n ln n law sets cache warmup, random test coverage,")
    print("  and how many samples it takes to see every category at least once.")

    _svg(os.path.join(outdir, "coupon_collector.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'coupon_collector.svg')}")


def _svg(path, w=760, h=380):
    n = 50
    et = expected_time(n)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="monospace">',
        f'<rect width="{w}" height="{h}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'The coupon collector: the last coupons dominate the wait</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'n = {n} coupons; expected draws to collect k of them (left), '
        f'completion probability (right)</text>',
    ]

    # left: expected draws to have collected k coupons, E to reach k = n*(H_n - H_{n-k})
    lx0, lx1 = 55, w // 2 - 20
    ly0, ly1 = h - 50, 60
    kmax, tmax = n, et

    def LX(k):
        return lx0 + k / kmax * (lx1 - lx0)

    def LY(t):
        return ly0 - t / tmax * (ly0 - ly1)

    parts.append(f'<line x1="{lx0}" y1="{ly0}" x2="{lx1}" y2="{ly0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{lx0}" y1="{ly0}" x2="{lx0}" y2="{ly1}" stroke="#8b949e" stroke-width="1.2"/>')
    curve = " ".join(f"{LX(k):.1f},{LY(expected_partial(n, k)):.1f}" for k in range(0, n + 1))
    parts.append(f'<polyline points="{curve}" fill="none" stroke="#4dabf7" stroke-width="2.5"/>')
    # mark the knee: the sharp rise near k=n
    parts.append(f'<line x1="{LX(n)}" y1="{ly1}" x2="{LX(n)}" y2="{ly0}" stroke="#ff6b6b" '
                 f'stroke-width="1" stroke-dasharray="3 3"/>')
    parts.append(f'<text x="{LX(n)-4:.1f}" y="{ly1+10:.1f}" fill="#ff6b6b" font-size="9" '
                 f'text-anchor="end">last coupon costs ~n</text>')
    parts.append(f'<text x="{(lx0+lx1)/2:.1f}" y="{ly0+22:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">coupons collected k</text>')
    parts.append(f'<text x="{lx0-8:.1f}" y="{ly1-4:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="end">draws</text>')

    # right: completion-probability CDF P(T <= t)
    rx0, rx1 = w // 2 + 40, w - 40
    ry0, ry1 = h - 50, 60
    t_hi = int(2.2 * et)

    def RX(t):
        return rx0 + t / t_hi * (rx1 - rx0)

    def RY(p):
        return ry0 - p * (ry0 - ry1)

    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx1}" y2="{ry0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx0}" y2="{ry1}" stroke="#8b949e" stroke-width="1.2"/>')
    cdf = " ".join(f"{RX(t):.1f},{RY(prob_complete_by(n, t)):.1f}"
                   for t in range(0, t_hi + 1, 5))
    parts.append(f'<polyline points="{cdf}" fill="none" stroke="#06d6a0" stroke-width="2.5"/>')
    # mark E[T]
    parts.append(f'<line x1="{RX(et):.1f}" y1="{ry1}" x2="{RX(et):.1f}" y2="{ry0}" '
                 f'stroke="#ffd43b" stroke-width="1" stroke-dasharray="3 3"/>')
    parts.append(f'<text x="{RX(et)+4:.1f}" y="{ry1+10:.1f}" fill="#ffd43b" font-size="9">'
                 f'E[T]={et:.0f}</text>')
    for p in (0.0, 0.5, 1.0):
        parts.append(f'<text x="{rx0-6:.1f}" y="{RY(p)+3:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="end">{p:.1f}</text>')
    parts.append(f'<text x="{(rx0+rx1)/2:.1f}" y="{ry0+22:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">draws t -> P(complete)</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
