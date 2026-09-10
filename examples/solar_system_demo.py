"""Demo: integrate the real solar system and recover Kepler's third law.

Measures each planet's orbital period by integrating Sun+planet, compares to the
analytic Kepler period and reality, and confirms T^2 proportional to a^3. Then
renders the inner planets' orbits to SVG.

    python examples/solar_system_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from solar_system import build, kepler_period, PLANETS  # noqa: E402
from render_svg import render  # noqa: E402

# real sidereal periods (years) for the reality check
REAL_PERIOD = {
    "Mercury": 0.2408, "Venus": 0.6152, "Earth": 1.0000, "Mars": 1.8808,
    "Jupiter": 11.862, "Saturn": 29.457, "Uranus": 84.021, "Neptune": 164.79,
}


def measure_period(planet):
    b = build([planet], include_sun=True)
    dt = kepler_period(PLANETS[planet][1]) / 2000.0

    def ang():
        return math.atan2(b.pos[1][1] - b.pos[0][1], b.pos[1][0] - b.pos[0][0])

    prev = ang()
    t = 0.0
    for _ in range(6000):
        b.step("verlet", dt)
        t += dt
        a = ang()
        if prev < 0 and a >= 0 and abs(a) < 1.0:
            return t
        prev = a
    return float("nan")


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Solar system in AU / years / solar masses (G = 4*pi^2)\n")
    print(f"{'planet':<9}{'a [AU]':>9}{'T measured':>12}{'T Kepler':>11}"
          f"{'T real':>10}{'T^2/a^3':>10}")
    print("-" * 61)
    for name, (m, a, e) in PLANETS.items():
        T = measure_period(name)
        Tk = kepler_period(a)
        ratio = T * T / a ** 3
        print(f"{name:<9}{a:>9.3f}{T:>12.4f}{Tk:>11.4f}"
              f"{REAL_PERIOD[name]:>10.4f}{ratio:>10.4f}")

    print("\nT^2/a^3 is constant across all planets -- that constant is 1 in these")
    print("units, which IS Kepler's third law. It falls straight out of Newtonian")
    print("gravity + a symplectic integrator, no fitting.")

    # render inner-planet orbits
    inner = ["Mercury", "Venus", "Earth", "Mars"]
    b = build(inner, include_sun=True)
    traj = b.record("verlet", kepler_period(PLANETS["Mars"][1]) / 4000,
                    8000, sample_every=8)
    path = os.path.join(outdir, "inner_planets.svg")
    render(traj, path, plane="xy", title="Inner solar system",
           subtitle="Mercury, Venus, Earth, Mars (+ Sun), 2 Mars years, verlet")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
