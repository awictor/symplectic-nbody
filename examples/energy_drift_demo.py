"""Demo: watch energy drift diverge between integrators.

Runs the eccentric two-body problem with each integrator and prints an ASCII
table plus a tiny sparkline of the relative energy error. No dependencies.

    python examples/energy_drift_demo.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from systems import two_body_eccentric  # noqa: E402

_BARS = " .:-=+*#@"


def sparkline(values):
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    return "".join(_BARS[min(8, int((v - lo) / span * 8))] for v in values)


def run(method, dt=0.01, steps=40000, samples=60):
    sys_ = two_body_eccentric(e=0.7)
    e0 = sys_.total_energy()
    every = max(1, steps // samples)
    errs = []
    for _t, e, _L in sys_.run(method, dt, steps, sample_every=every):
        errs.append((e - e0) / e0)
    max_abs = max(abs(x) for x in errs)
    net = errs[-1]
    return errs, max_abs, net


def main():
    print("Eccentric two-body (e=0.7), dt=0.01, 40k steps, G=1\n")
    print(f"{'method':<14}{'max |dE/E|':<16}{'net dE/E':<16}drift shape")
    print("-" * 72)
    for method in ("verlet", "forest_ruth", "rk4"):
        errs, mx, net = run(method)
        print(f"{method:<14}{mx:<16.3e}{net:<16.3e}{sparkline(errs)}")
    print("\nSymplectic methods (verlet, forest_ruth) oscillate around 0.")
    print("RK4's error walks off in one direction -- that's secular energy drift.")


if __name__ == "__main__":
    main()
