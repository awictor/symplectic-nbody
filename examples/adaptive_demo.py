"""Demo: adaptive Dormand-Prince RK45 vs fixed-step RK4 on an eccentric orbit.

Shows (a) how the step size collapses at pericenter and expands at apocenter,
and (b) how many fewer force evaluations adaptive stepping needs to reach the
same end-of-orbit accuracy.

    python examples/adaptive_demo.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from systems import two_body_eccentric  # noqa: E402
from adaptive import DormandPrince  # noqa: E402
from integrators import rk4  # noqa: E402

_BARS = " .:-=+*#@"


def sparkline(values):
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    return "".join(_BARS[min(8, int((v - lo) / span * 8))] for v in values)


def period(a=1.0, mu=2.0):
    return 2.0 * math.pi * math.sqrt(a ** 3 / mu)


def main():
    T = period()
    print(f"Eccentric two-body (e=0.7), one full period T={T:.4f}, G=1\n")

    # Reference "truth" from a very tight adaptive run.
    s0 = two_body_eccentric(e=0.7)
    ref = DormandPrince(s0.accel, s0.n, rtol=1e-13, atol=1e-15, h_init=1e-4)
    ref_pos, _ = ref.integrate(s0.pos, s0.vel, T)

    def end_err(pos):
        return max(abs(pos[i][k] - ref_pos[i][k])
                   for i in range(len(pos)) for k in range(3))

    # Adaptive run, recording step sizes over the orbit.
    s1 = two_body_eccentric(e=0.7)
    steps, last = [], [0.0]

    def sample(t, p, v):
        steps.append(t - last[0]); last[0] = t

    dp = DormandPrince(s1.accel, s1.n, rtol=1e-8, atol=1e-11, h_init=1e-3)
    p_ad, _ = dp.integrate(s1.pos, s1.vel, T, on_sample=sample)
    err_ad = end_err(p_ad)

    print("adaptive step size over the orbit (small=pericenter, large=apocenter):")
    # downsample to ~60 columns for the sparkline
    stride = max(1, len(steps) // 60)
    print("  " + sparkline(steps[::stride]))
    print(f"  steps accepted={dp.n_accepted}  rejected={dp.n_rejected}  "
          f"h_min={min(steps):.2e}  h_max={max(steps):.2e}  ratio={max(steps)/min(steps):.0f}x\n")

    # Fixed RK4: find the coarsest step count that matches adaptive accuracy.
    print(f"{'method':<24}{'force evals':>14}{'end error':>14}")
    print("-" * 52)
    print(f"{'adaptive DP45':<24}{dp.n_feval:>14}{err_ad:>14.2e}")
    for n in (2000, 4000, 8000, 16000, 32000, 64000):
        s2 = two_body_eccentric(e=0.7)
        dt = T / n
        pos, vel = [list(p) for p in s2.pos], [list(v) for v in s2.vel]
        for _ in range(n):
            pos, vel = rk4(pos, vel, s2.accel, dt)
        e = end_err(pos)
        if e <= err_ad:
            print(f"{'fixed RK4 (' + str(n) + ' steps)':<24}{n*4:>14}{e:>14.2e}")
            print(f"\nadaptive reaches the same accuracy with "
                  f"{(n*4)/dp.n_feval:.1f}x fewer force evaluations.")
            break
    else:
        print("fixed RK4 did not match adaptive accuracy within 64000 steps.")


if __name__ == "__main__":
    main()
