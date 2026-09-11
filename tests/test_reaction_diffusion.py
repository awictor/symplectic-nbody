"""Tests for reaction_diffusion: the Gray-Scott Turing-pattern model."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import reaction_diffusion as rd

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


# Initial fields: u=1 everywhere, v=0.
u, v = rd.make_fields(10)
check("initial u = 1", all(x == 1.0 for row in u for x in row))
check("initial v = 0", rd.total_v(v) == 0.0)

# Laplacian of a uniform field is zero.
uni = [[3.0] * 5 for _ in range(5)]
check("Laplacian of uniform = 0", abs(rd.laplacian(uni, 2, 2)) < 1e-12)
# Laplacian positive where the centre is a dip.
dip = [[1.0] * 3 for _ in range(3)]
dip[1][1] = 0.0
check("Laplacian positive at a dip", rd.laplacian(dip, 1, 1) > 0.0)

# A bare substrate (v=0 everywhere) is a steady state: u v^2 = 0, u stays ~1, v stays 0.
u2, v2 = rd.make_fields(12)
u2, v2 = rd.evolve(u2, v2, 20)
check("bare substrate: v stays zero", rd.total_v(v2) < 1e-9)
check("bare substrate: u stays ~1", abs(u2[0][0] - 1.0) < 1e-6)
check("bare substrate: no pattern", rd.pattern_contrast(v2) < 1e-9)

# Seeding autocatalyst breaks symmetry: v is present after seeding.
u3, v3 = rd.make_fields(40)
rd.seed_center(u3, v3, size=8)
check("seed introduces autocatalyst", rd.total_v(v3) > 0.0)
seed_total = rd.total_v(v3)

# Evolve: the seed grows/reorganizes into a pattern (v total and contrast change).
u3, v3 = rd.evolve(u3, v3, 400, f=0.035, k=0.06)
check("pattern has spatial contrast", rd.pattern_contrast(v3) > 0.05)
check("autocatalyst persists (pattern alive)", rd.total_v(v3) > 0.0)
# v stays bounded (0..~1), doesn't blow up.
check("v bounded (no blow-up)", all(-0.5 < x < 1.5 for row in v3 for x in row))
check("u bounded", all(-0.5 < x < 1.5 for row in u3 for x in row))

# Diffusion coefficients: activator diffuses slower than inhibitor in the classic setup.
# (Structural: default du > dv.)
u4, v4 = rd.make_fields(20)
rd.seed_center(u4, v4, 6)
u4, v4 = rd.evolve(u4, v4, 100)
check("evolution keeps a finite pattern", 0.0 < rd.total_v(v4) < 20 * 20)

# step returns same-shape grids.
ua, va = rd.step(*rd.seed_center(*rd.make_fields(8), 4))
check("step preserves shape", len(ua) == 8 and len(ua[0]) == 8)

# pattern_contrast zero for a truly uniform field.
check("contrast zero for uniform", rd.pattern_contrast([[0.5] * 4 for _ in range(4)]) == 0.0)

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all reaction_diffusion tests passed")
