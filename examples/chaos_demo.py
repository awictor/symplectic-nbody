"""Demo: chaos and the largest Lyapunov exponent in the three-body problem.

Shows the defining feature of chaos: two trajectories starting 1e-9 apart in one
coordinate diverge to order-unity separation, on an exponential clock set by the
Lyapunov exponent. Contrasts a chaotic system (pythagorean) with a regular one,
and renders both nearby trajectories to SVG so you can see them peel apart.

    python examples/chaos_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from systems import pythagorean, two_body_circular  # noqa: E402
from lyapunov import largest_lyapunov, lyapunov_time  # noqa: E402
from nbody import NBody  # noqa: E402
from render_svg import render  # noqa: E402

_BARS = " .:-=+*#@"


def sparkline(vals):
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    return "".join(_BARS[min(8, int((v - lo) / span * 8))] for v in vals)


def _clone(s, dx=0.0):
    t = NBody(masses=list(s.m), pos=[list(p) for p in s.pos],
              vel=[list(v) for v in s.vel], G=s.G, softening=math.sqrt(s.soft2))
    t.pos[0][0] += dx
    return t


def divergence_series(system, dt, steps, sample):
    ref = _clone(system)
    twin = _clone(system, dx=1e-9)
    ts, ds = [], []
    for i in range(steps):
        ref.step("forest_ruth", dt)
        twin.step("forest_ruth", dt)
        if i % sample == 0:
            d = 0.0
            for b in range(ref.n):
                for k in range(3):
                    d += (ref.pos[b][k] - twin.pos[b][k]) ** 2
            ts.append(i * dt)
            ds.append(math.sqrt(d) + 1e-18)
    return ts, ds


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Chaos in the three-body problem: the largest Lyapunov exponent\n")

    lam_c = largest_lyapunov(pythagorean(), dt=0.0005, total_time=100)
    lam_r = largest_lyapunov(two_body_circular(), dt=0.002, total_time=200)
    print(f"  pythagorean 3-body : lambda = {lam_c:.3f}   "
          f"Lyapunov time ~ {1/lam_c:.1f} time units")
    print(f"  regular two-body   : lambda = {lam_r:.3f}   (decays toward 0 with T)")
    print(f"\n  => the chaotic system's exponent is {lam_c/lam_r:.0f}x larger.\n")

    ts, ds = divergence_series(pythagorean(), dt=0.0005, steps=60000, sample=500)
    print("separation of two trajectories started 1e-9 apart (log scale):")
    logs = [math.log10(d) for d in ds]
    print("  " + sparkline(logs))
    print(f"  grew from 1e-9 to ~{ds[-1]:.1e} -- {math.log10(ds[-1]/ds[0]):.0f} orders "
          f"of magnitude. That's why the long-term 3-body problem is unpredictable.")

    # render the two nearly-identical initial conditions peeling apart
    ref = _clone(pythagorean())
    twin = _clone(pythagorean(), dx=1e-6)
    traj = [[tuple(ref.pos[0])], [tuple(twin.pos[0])]]
    for i in range(40000):
        ref.step("forest_ruth", 0.0005)
        twin.step("forest_ruth", 0.0005)
        if i % 40 == 0:
            traj[0].append(tuple(ref.pos[0]))
            traj[1].append(tuple(twin.pos[0]))
    path = os.path.join(outdir, "chaos_divergence.svg")
    render(traj, path, plane="xy", title="Sensitive dependence (body 1 of 3)",
           subtitle="two runs, initial positions 1e-6 apart, tracked over time")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
