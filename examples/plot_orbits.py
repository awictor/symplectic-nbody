"""Demo: render orbit trajectories to standalone SVG files (zero dependencies).

    python examples/plot_orbits.py [output_dir]

Writes:
  figure_eight.svg  -- the Chenciner-Montgomery three-body choreography
  eccentric.svg     -- an e=0.7 two-body orbit (watch the ellipse close)
  pythagorean.svg   -- Burrau's chaotic 3-4-5 problem
Open any of them in a browser.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from systems import figure_eight, two_body_eccentric, pythagorean  # noqa: E402
from render_svg import render  # noqa: E402
from nbody import NBody  # noqa: E402


def pythagorean_softened():
    """Pythagorean 3-body with heavier softening so the fixed-step integrator
    stays energy-stable through the close encounters -- gives a clean plot."""
    p = pythagorean()
    return NBody(masses=p.m, pos=p.pos, vel=p.vel, softening=0.1)


def _drift(system, method, dt, steps):
    e0 = system.total_energy()
    traj = system.record(method, dt, steps, sample_every=max(1, steps // 2000))
    net = abs((system.total_energy() - e0) / e0)
    return traj, net


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    jobs = [
        ("figure_eight", figure_eight(), "forest_ruth", 0.002, 3140,
         "Figure-eight choreography", "3 equal masses, one shared orbit"),
        ("eccentric", two_body_eccentric(e=0.7), "verlet", 0.005, 4000,
         "Eccentric two-body (e=0.7)", "started at apoapsis, G=1"),
        ("pythagorean", pythagorean_softened(), "forest_ruth", 0.0005, 120000,
         "Burrau pythagorean 3-body", "masses 3-4-5, chaotic close encounters"),
    ]

    for name, system, method, dt, steps, title, sub in jobs:
        traj, net = _drift(system, method, dt, steps)
        subtitle = f"{sub}   |   {method}, dt={dt}, {steps} steps, net dE/E={net:.1e}"
        path = os.path.join(outdir, f"{name}.svg")
        render(traj, path, plane="xy", title=title, subtitle=subtitle)
        print(f"wrote {path}  ({len(traj)} bodies, {len(traj[0])} samples, drift {net:.1e})")


if __name__ == "__main__":
    main()
