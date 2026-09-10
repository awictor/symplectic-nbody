"""Demo: Mercury's perihelion precession -- the first triumph of general relativity.

Prints the famous 43 arcsec/century (from the 1PN closed form), shows the
numeric integration reproducing the analytic advance, and renders a "rosette":
an orbit with the GR effect amplified so the slowly-turning ellipse is visible.

    python examples/precession_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from relativity import (  # noqa: E402
    analytic_precession_per_orbit, precession_per_orbit,
    mercury_precession_arcsec_per_century, accel_1pn, _rk4_step,
    C_LIGHT, GM_SUN, ARCSEC_PER_RAD,
)
from render_svg import render  # noqa: E402

MERCURY_A, MERCURY_E = 0.387098, 0.205630


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Perihelion precession of Mercury (1PN general relativity)\n")
    val = mercury_precession_arcsec_per_century()
    print(f"  analytic advance at the real speed of light: {val:.2f} arcsec/century")
    print(f"  observed / GR-predicted value:               ~43 arcsec/century\n")

    print("numeric integration reproduces the closed form 6*pi*GM/(c^2 a(1-e^2)):")
    print(f"  {'c factor':>10}{'numeric/orbit':>16}{'analytic/orbit':>16}{'ratio':>9}")
    print("  " + "-" * 51)
    for cfac in (300, 400, 600):
        c = C_LIGHT / cfac
        num = precession_per_orbit(MERCURY_A, MERCURY_E, GM_SUN, c,
                                   orbits=20, steps_per_orbit=12000)
        ana = analytic_precession_per_orbit(MERCURY_A, MERCURY_E, GM_SUN, c)
        print(f"  c/{cfac:<8}{num:>16.6e}{ana:>16.6e}{num/ana:>9.4f}")

    # rosette: amplify GR hard so the ellipse visibly rotates
    c = C_LIGHT / 8000.0
    a, e = 1.0, 0.6
    r_aph = a * (1 + e)
    v_aph = math.sqrt(GM_SUN * (2.0 / r_aph - 1.0 / a))
    state = [r_aph, 0.0, 0.0, v_aph]
    period = 2 * math.pi * math.sqrt(a ** 3 / GM_SUN)
    dt = period / 4000
    pts = [(state[0], state[1], 0.0)]
    for _ in range(4000 * 8):
        state = _rk4_step(state, dt, GM_SUN, c)
        pts.append((state[0], state[1], 0.0))
    path = os.path.join(outdir, "precession_rosette.svg")
    render([pts], path, plane="xy", title="Relativistic precession (rosette)",
           subtitle="GR amplified ~8000x: the ellipse slowly rotates instead of closing")
    print(f"\nwrote {path}")
    print("At the true speed of light the same rotation is a mere 43 arcsec/century --")
    print("undetectable in one orbit, unmistakable over a century of Mercury's.")


if __name__ == "__main__":
    main()
