"""Tests for percolation: connectivity threshold and cluster analysis."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import percolation as pc

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


# Occupation fraction ~ p for a large lattice.
grid = pc.occupy_lattice(50, 0.6, seed=1)
occ = sum(sum(row) for row in grid) / (50 * 50)
check("occupation fraction ~ p", abs(occ - 0.6) < 0.05)
check("reproducible with same seed",
      pc.occupy_lattice(20, 0.5, seed=7) == pc.occupy_lattice(20, 0.5, seed=7))
check("different seeds differ",
      pc.occupy_lattice(20, 0.5, seed=1) != pc.occupy_lattice(20, 0.5, seed=2))

# Empty lattice: no clusters, no spanning.
empty = pc.occupy_lattice(20, 0.0)
check("empty lattice does not span", not pc.spans(empty))
check("empty lattice zero largest cluster", pc.largest_cluster_fraction(empty) == 0.0)

# Full lattice: one giant cluster spanning, fraction 1.
full = pc.occupy_lattice(20, 1.0)
check("full lattice spans", pc.spans(full))
check("full lattice largest cluster = 1", abs(pc.largest_cluster_fraction(full) - 1.0) < 1e-9)

# A hand-built vertical stripe spans; a horizontal one does not.
n = 5
stripe = [[c == 2 for c in range(n)] for _ in range(n)]     # column 2 occupied
check("vertical stripe spans top-bottom", pc.spans(stripe))
hstripe = [[r == 2 for c in range(n)] for r in range(n)]    # row 2 occupied
check("horizontal stripe does not span", not pc.spans(hstripe))

# Largest-cluster fraction rises with p.
f_low = pc.largest_cluster_fraction(pc.occupy_lattice(40, 0.4, seed=3))
f_high = pc.largest_cluster_fraction(pc.occupy_lattice(40, 0.75, seed=3))
check("largest cluster grows with p", f_high > f_low)
# Well above threshold, the giant cluster dominates.
check("above threshold giant cluster large", f_high > 0.5)
# Well below threshold, clusters stay small.
check("below threshold clusters small",
      pc.largest_cluster_fraction(pc.occupy_lattice(40, 0.3, seed=3)) < 0.2)

# Percolation probability: near zero well below p_c, near one well above.
pp_low = pc.percolation_probability(30, 0.45, trials=15)
pp_high = pc.percolation_probability(30, 0.75, trials=15)
check("rarely percolates below threshold", pp_low < 0.3)
check("almost always percolates above threshold", pp_high > 0.8)
check("percolation probability rises through threshold", pp_high > pp_low)
# Right around p_c ~ 0.59 it is intermediate.
pp_c = pc.percolation_probability(30, pc.SQUARE_SITE_THRESHOLD, trials=20)
check("intermediate probability at p_c", 0.2 < pp_c < 0.9)

# Threshold constant is ~0.59.
check("square-lattice threshold ~0.59", abs(pc.SQUARE_SITE_THRESHOLD - 0.5927) < 0.001)

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all percolation tests passed")
