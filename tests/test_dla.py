"""Tests for dla.py -- diffusion-limited aggregation.

Self-running: prints PASS/FAIL per check, exits 1 if any fail.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import dla  # noqa: E402

failed = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'} {name}")
    if not cond:
        failed.append(name)


def approx(a, b, tol):
    return abs(a - b) <= tol


# --- grow: basic structure --------------------------------------------------
c = dla.grow(300, seed=5)
check("cluster has requested size", len(c) == 300)
check("seed at origin is present", (0, 0) in c)
check("all cells are integer pairs", all(isinstance(x, int) and isinstance(y, int)
                                         for x, y in c))
check("no duplicate cells (it's a set)", len(c) == len(set(c)))

# every cell (except the seed) must touch the cluster via a 4-neighbour: DLA is connected
seed = (0, 0)
connected = True
for cell in c:
    if cell == seed:
        continue
    x, y = cell
    if not any((x + dx, y + dy) in c for dx, dy in dla._STEPS):
        connected = False
        break
check("every stuck cell is adjacent to another (connected)", connected)

# determinism: same seed -> same cluster
c2 = dla.grow(300, seed=5)
check("same seed reproduces the cluster", c == c2)
c3 = dla.grow(300, seed=6)
check("different seed gives a different cluster", c != c3)

# --- radius of gyration -----------------------------------------------------
# a single seed has Rg 0
one = dla.grow(1, seed=1)
check("single-cell cluster has Rg 0", approx(dla.radius_of_gyration(one), 0.0, 1e-9))
check("bigger cluster has positive Rg", dla.radius_of_gyration(c) > 1.0)

# --- mass within radius -----------------------------------------------------
cx, cy = dla.center_of_mass(c)
r_out = max(math.hypot(a - cx, b - cy) for a, b in c)
check("mass within full radius is the whole cluster", dla.mass_within(c, r_out + 1) == len(c))
check("mass within radius is monotonic increasing",
      dla.mass_within(c, r_out / 4) <= dla.mass_within(c, r_out / 2) <= dla.mass_within(c, r_out))
check("mass within zero radius is small", dla.mass_within(c, 0.0) <= 2)

# --- fractal dimension ------------------------------------------------------
# a DLA cluster is a fractal: D well below the space dimension 2, in the ~1.5-1.9 band.
big = dla.grow(1500, seed=3)
D, pts = dla.fractal_dimension(big)
check("fractal dimension is sub-plane (D < 1.95)", D < 1.95)
check("fractal dimension exceeds a filament (D > 1.35)", D > 1.35)
check("fractal dimension near the 2D DLA value ~1.71", approx(D, 1.71, 0.35))
check("scaling points returned", len(pts) >= 4)

# a solid disc (every lattice cell in a radius) must give D ~ 2, distinguishing it from DLA
R = 20
disc = {(x, y) for x in range(-R, R + 1) for y in range(-R, R + 1)
        if x * x + y * y <= R * R}
Dd, _ = dla.fractal_dimension(disc)
check("a filled disc scales as D ~ 2 (not fractal)", approx(Dd, 2.0, 0.2))
check("DLA is sparser than a disc (D_dla < D_disc)", D < Dd)

# --- bounds -----------------------------------------------------------------
xmin, ymin, xmax, ymax = dla.bounds(c)
check("bounds contain all cells",
      all(xmin <= x <= xmax and ymin <= y <= ymax for x, y in c))
check("bounds include the origin seed", xmin <= 0 <= xmax and ymin <= 0 <= ymax)


if failed:
    print(f"\n{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("\nall dla tests passed")
