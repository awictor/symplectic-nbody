"""Demo: Barnes-Hut O(N log N) vs direct O(N^2) force evaluation.

Times one full force evaluation for growing N with both methods and prints the
speedup plus the empirical scaling exponent. No dependencies (uses time.perf_counter).

    python examples/scaling_benchmark.py
"""

import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from systems import plummer_sphere  # noqa: E402
from barnes_hut import BarnesHut  # noqa: E402


def time_call(fn, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    Ns = [200, 400, 800, 1600, 3200]
    print("Force-evaluation time: direct O(N^2) vs Barnes-Hut O(N log N), theta=0.5\n")
    print(f"{'N':>6}{'direct (ms)':>15}{'bh (ms)':>12}{'speedup':>10}")
    print("-" * 45)
    direct_t, bh_t = [], []
    for n in Ns:
        s = plummer_sphere(n=n, seed=3)
        bh = BarnesHut(G=s.G, theta=0.5, softening=(s.soft2 ** 0.5))
        td = time_call(lambda: s.accel(s.pos))
        tb = time_call(lambda: bh.accel(s.pos, s.m))
        direct_t.append(td); bh_t.append(tb)
        print(f"{n:>6}{td * 1e3:>15.2f}{tb * 1e3:>12.2f}{td / tb:>9.1f}x")

    def exponent(ts):
        # slope of log(t) vs log(N) = empirical scaling exponent
        lx = [math.log(n) for n in Ns]
        ly = [math.log(t) for t in ts]
        mx, my = sum(lx) / len(lx), sum(ly) / len(ly)
        num = sum((lx[i] - mx) * (ly[i] - my) for i in range(len(lx)))
        den = sum((lx[i] - mx) ** 2 for i in range(len(lx)))
        return num / den

    print(f"\nempirical scaling exponent  direct ~ N^{exponent(direct_t):.2f}"
          f"   barnes-hut ~ N^{exponent(bh_t):.2f}")
    print("direct sits at ~2.0 (quadratic). barnes-hut is clearly sub-quadratic;")
    print("its exponent falls toward the N log N asymptote as N grows and the")
    print("per-call tree-build overhead is amortized over more force terms.")


if __name__ == "__main__":
    main()
