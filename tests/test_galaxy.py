"""Disk-galaxy and collision tests.

Claims checked:
  1. A disk is cold and in equilibrium: an ISOLATED galaxy's tracers stay near
     their starting radii (circular orbits don't fly apart).
  2. Tracers are massless and the two central masses carry all the mass.
  3. A close two-galaxy passage draws a significant fraction of the cold disk
     material out past its initial radius -- tidal tails form.
  4. The build is deterministic (fixed seed -> identical initial state).
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from galaxy import make_disk, two_galaxy_encounter, centers_of  # noqa: E402
from nbody import NBody  # noqa: E402
from barnes_hut import BarnesHut  # noqa: E402
from integrators import velocity_verlet  # noqa: E402


def test_isolated_disk_is_stable():
    m, p, v = make_disk([0, 0, 0], [0, 0, 0], m_center=1.0, n_ring=200,
                        r_in=0.3, r_out=1.5, seed=5)
    g = NBody(masses=m, pos=p, vel=v, G=1.0, softening=0.05)
    # initial radii from the (stationary) center
    r0 = [math.dist(p[i], p[0]) for i in range(1, len(p))]
    for _ in range(300):
        g.pos, g.vel = velocity_verlet(g.pos, g.vel, g.accel, 0.02)
    r1 = [math.dist(g.pos[i], g.pos[0]) for i in range(1, len(g.pos))]
    # mean radius should barely change for a cold circular disk
    mean0 = sum(r0) / len(r0)
    mean1 = sum(r1) / len(r1)
    assert abs(mean1 - mean0) / mean0 < 0.15, (
        f"isolated disk drifted: mean r {mean0:.3f} -> {mean1:.3f}")


def test_tracers_are_massless():
    m, p, v = make_disk([0, 0, 0], [0, 0, 0], m_center=2.0, n_ring=50)
    assert m[0] == 2.0
    assert all(mi == 0.0 for mi in m[1:])


def test_two_centers_only_massive_bodies():
    sysn = two_galaxy_encounter(n_ring=100)
    heavy = [i for i, mi in enumerate(sysn.m) if mi > 0.0]
    assert len(heavy) == 2, f"expected 2 central masses, got {len(heavy)}"


def test_collision_forms_tails():
    sysn = two_galaxy_encounter(n_ring=300, inclination=0.4)
    cA, cB = centers_of(sysn)
    bh = BarnesHut(G=sysn.G, theta=0.6, softening=(sysn.soft2 ** 0.5))
    pos, vel = sysn.pos, sysn.vel
    for _ in range(500):
        pos, vel = velocity_verlet(pos, vel, lambda p: bh.accel(p, sysn.m), 0.02)
    flung = sum(1 for i in range(sysn.n) if sysn.m[i] == 0.0 and
                min(math.dist(pos[i], pos[cA]), math.dist(pos[i], pos[cB])) > 2.5)
    # a real close passage strips a substantial fraction of the (r<1.5) disk
    assert flung > 50, f"expected tidal tails, only {flung} tracers stripped"


def test_deterministic():
    a = two_galaxy_encounter(n_ring=80)
    b = two_galaxy_encounter(n_ring=80)
    assert a.pos[10] == b.pos[10] and a.vel[50] == b.vel[50]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
